import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from matplotlib.widgets import Slider, Button

from qcodes.dataset import experiments, initialise_or_create_database_at
from qcodes.dataset.data_set import load_by_id

logger = logging.getLogger(__name__)


def _h5_group_to_dict(group):
    """Recursively turn an h5py Group into a plain nested dict of
    arrays/scalars, so the metadata survives after the file is closed
    instead of holding a live h5py handle open."""
    out = {}
    for key, item in group.items():
        out[key] = _h5_group_to_dict(item) if isinstance(item, h5py.Group) else item[()]
    return out


class H5DataReader:
    """Reads .h5 measurement files into arrays/dicts ready to hand to
    PlotterBase's subclasses (Plot1D / MultiDimPlotter).

    Covers both the per-(current,freq) files a multi-dimensional sweep
    splits across (see e.g.
    `measurement_scripts/spectro_awgPump_sweep_variable_ranges_simpNOCompOnly_powSlope_sweep_flux.py`)
    and the single-file 'metadata' block every `BaseMeasurement` subclass
    writes (see `measurement_scripts/base_measurement.py`).
    """

    def __init__(self, data_path):
        self.data_path = data_path

    def load_fs_fp_map(self, basename, save=False):
        """Stitch together every `<basename>_current<i>_freq<j>.h5` file
        into one array, shape (current, pump_freq, signal_freq, pump_amp,
        Sij)."""
        data = []
        i = 0
        loop = True
        while loop:
            j = 0
            while True:
                try:
                    fname = self.data_path + basename + "_current{}_freq{}.h5".format(i, j)
                    f = h5py.File(fname, 'r')
                    if j == 0:
                        data += [[]]
                    data[-1] += [np.array(f['data'])]
                    j += 1
                except FileNotFoundError:
                    if j > 0:
                        break
                    elif j == 0:
                        if i > 0:
                            loop = False
                            break
                        elif i == 0:
                            raise FileNotFoundError(fname)
            i += 1
        data = np.array(data)  # shape (current, pump_freq, pump_amp, signal_freq, Sij)
        data = np.einsum('cijk...->cikj...', data)  # shape (current, pump_freq, signal_freq, pump_amp, Sij)
        if save:
            np.save(self.data_path + basename, data)
        return data

    def read_metadata(self, filename):
        """Open `filename` and return its 'metadata' group (instrument
        configs, measurement params, circuit path - see
        `BaseMeasurement.save_measurement_info`) as a plain nested dict."""
        with h5py.File(self.data_path + filename, 'r') as f:
            return _h5_group_to_dict(f['metadata'])

    def read_data(self, filename, dataset='data'):
        """Open `filename` and return one dataset (default 'data') as a
        numpy array - the direct path for a single-run .h5 with no
        per-(current,freq) splitting."""
        with h5py.File(self.data_path + filename, 'r') as f:
            return np.array(f[dataset])


class QCodesDatabaseReader:
    """Reads a qcodes sqlite database - the .db file
    `measurement_scripts/qcodes_utils/measurement_run.py` writes every run
    into - returning qcodes DataSets/DataFrames/xarray Datasets ready to
    hand to PlotterBase's subclasses (Plot1D / MultiDimPlotter)."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        initialise_or_create_database_at(self.db_path)

    def list_runs(self):
        """List every run in this database as one dict per run - run_id,
        experiment/sample name, run name, timestamp, completed - so you can
        see what's actually there before loading anything."""
        runs = []
        for exp in experiments():
            for ds in exp.data_sets():
                runs.append({
                    "run_id": ds.run_id,
                    "experiment_name": exp.name,
                    "sample_name": exp.sample_name,
                    "name": ds.name,
                    "timestamp": ds.run_timestamp(),
                    "completed": ds.completed,
                })
        return runs

    def load_run(self, run_id):
        """Load one run by its (database-local) `run_id` - see
        `list_runs` to find it - as a qcodes `DataSet`."""
        return load_by_id(run_id)

    def load_latest_run(self, experiment_name, sample_name=None):
        """Load the most recently captured run of `experiment_name` as a
        qcodes `DataSet` - "give me whatever I last measured under this
        experiment", without having to know its run_id. Pass `sample_name`
        too if more than one sample shares that experiment name."""
        matches = [
            exp for exp in experiments()
            if exp.name == experiment_name
            and (sample_name is None or exp.sample_name == sample_name)
        ]
        if not matches:
            raise ValueError(f"no experiment named {experiment_name!r} in {self.db_path}")
        latest_exp = matches[-1]
        data_sets = latest_exp.data_sets()
        if not data_sets:
            raise ValueError(f"experiment {experiment_name!r} in {self.db_path} has no runs")
        return data_sets[-1]

    def load_run_dataframe(self, run_id=None, experiment_name=None, sample_name=None):
        """`load_run`/`load_latest_run`, returned as a pandas DataFrame -
        the quickest way to get a run's data into something you can
        `.plot()` or slice directly. Pass `run_id`, or `experiment_name`
        (+ optional `sample_name`) for the latest run of that experiment.

        Raises `NotImplementedError` for a run with more than one
        independent-parameter tree of different shapes - which every
        multi-instrument sweep in this repo except
        `vna_calib_slope_custom_meas` is (qcodes can't concat those into
        one flat DataFrame). For those, use `load_run`/`load_latest_run`
        directly and call `.get_parameter_data()` on the DataSet instead."""
        dataset = (
            self.load_run(run_id) if run_id is not None
            else self.load_latest_run(experiment_name, sample_name)
        )
        return dataset.to_pandas_dataframe()

    def load_run_xarray(self, run_id=None, experiment_name=None, sample_name=None):
        """Same as `load_run_dataframe`, returned as an xarray Dataset
        instead - convenient for multi-dimensional sweeps, since every
        setpoint axis stays labeled instead of being flattened into a
        DataFrame's multi-index."""
        dataset = (
            self.load_run(run_id) if run_id is not None
            else self.load_latest_run(experiment_name, sample_name)
        )
        return dataset.to_xarray_dataset()


class PlotterBase:
    """Base class for shared plotter functionality."""

    def _setup_optional_params(self, dataset_names, colors, linestyles, markers):
        """Setup optional parameters (dataset_names, colors, linestyles, markers)."""
        if dataset_names is None:
            dataset_names = [f'Dataset {i + 1}' for i in range(self.n_datasets)]
        self.dataset_names = dataset_names

        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, self.n_datasets)))
        self.colors = colors

        if linestyles is None:
            linestyles = ['-'] * self.n_datasets
        self.linestyles = linestyles

        if markers is None:
            markers = [''] * self.n_datasets
        self.markers = markers

    def _create_sliders(self):
        """Create sliders for non-plotted axes. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_sliders")

    def _make_slider_callback(self, axis):
        """Create a callback for slider changes."""
        def callback(val):
            self._on_slider_changed(axis, int(val))
        return callback

    def _on_slider_changed(self, axis, value):
        """Handle slider changes. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _on_slider_changed")

    def _update_sliders(self):
        """Recreate sliders for the new plotted axes."""
        # Remove old sliders and their axes
        for i, slider in list(self.sliders.items()):
            try:
                self.fig.delaxes(slider.ax)
            except:
                pass

        self.sliders.clear()
        self.slider_axes.clear()

        # Recreate sliders for new axes
        self._create_sliders()

        self.fig.canvas.draw_idle()

    def _get_global_min_max(self):
        """Calculate global min and max across all data in data_list."""
        global_min = min(np.amin(arr[~np.isnan(arr)]) for arr in self.data_list)
        global_max = max(np.amax(arr[~np.isnan(arr)]) for arr in self.data_list)
        return global_min, global_max


class Plot1D(PlotterBase):
    """1D plot viewer for multidimensional data - pure matplotlib with independent axis scrolling."""

    def __init__(self, data_list, x_axis, y_axis, labels, values, slices, initial_click_x_idx=0, initial_click_y_idx=0,
                 dataset_names=None, colors=None, linestyles=None, markers=None):
        """
        Initialize the 1D plotter.

        Parameters:
        -----------
        data_list : list of np.ndarray
            List of data arrays
        x_axis : int
            Index of the axis to plot on X initially
        y_axis : int
            Index of the axis that corresponds to Y values (for initial click)
        labels : list of str or None
            Names of all axes. If None, uses generic names (Axis 0, Axis 1, ...)
        values : list of list
            Values for each axis
        slices : dict
            Current slice indices for all dimensions
        initial_click_x_idx : int
            Initial X index from the click
        initial_click_y_idx : int
            Initial Y index from the click
        dataset_names : list of str, optional
            Names for each dataset. If None, uses 'Dataset 1', 'Dataset 2', ...
        colors : list of str, optional
            Colors for each line. If None, uses default colormap
        linestyles : list of str, optional
            Line styles for each dataset. If None, uses '-' (solid line)
        markers : list of str, optional
            Markers for each line. If None, uses no markers (empty string)
        """
        self.data_list = data_list
        self.labels = labels if labels is not None else [f'Axis {i}' for i in range(len(values))]
        self.values = values
        self.slices = slices.copy()
        self.ndim = len(self.labels)
        self.n_datasets = len(data_list)

        # ...existing code...
        self._setup_optional_params(dataset_names, colors, linestyles, markers)

        # Current state - current_x_axis is the axis being plotted
        self.current_x_axis = x_axis
        # Update slices based on the click
        self.slices[x_axis] = initial_click_x_idx
        self.slices[y_axis] = initial_click_y_idx

        # Calculate global min/max across all datasets (not just current slice)
        self.global_data_min, self.global_data_max = self._get_global_min_max()

        # Current y-axis range (will be updated by buttons)
        self.y_axis_min = self.global_data_min
        self.y_axis_max = self.global_data_max

        # Store zoom limits (for preserving zoom when using scrollers)
        self.zoom_xlim = None
        self.zoom_ylim = None

        # Create figure with matplotlib
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('1D Data Viewer', fontsize=14)

        # Create main axis for plotting
        self.ax_plot = self.fig.add_subplot(111)
        self.lines = []  # Store line objects for all datasets

        for i in range(self.n_datasets):
            line, = self.ax_plot.plot([], [], linestyle=self.linestyles[i % len(self.linestyles)],
                                       marker=self.markers[i % len(self.markers)], linewidth=2,
                                       label=self.dataset_names[i], color=self.colors[i % len(self.colors)])
            self.lines.append(line)

        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.legend()

        # Create sliders for ALL other axes (not the one being plotted)
        self.sliders = {}
        self.slider_axes = {}
        self._create_sliders()

        # Create axis cycling buttons positioned at same height as sliders with spacing
        self.ax_prev_axis = self.fig.add_axes([0.78, 0.16, 0.06, 0.04])
        self.btn_prev_axis = Button(self.ax_prev_axis, '← Axis')
        self.btn_prev_axis.on_clicked(self._on_prev_axis_clicked)

        self.ax_next_axis = self.fig.add_axes([0.78, 0.10, 0.06, 0.04])
        self.btn_next_axis = Button(self.ax_next_axis, 'Axis →')
        self.btn_next_axis.on_clicked(self._on_next_axis_clicked)

        # Create dataset rescale buttons stacked vertically on the right
        button_width = 0.08
        button_height = 0.04
        button_spacing = 0.01
        start_x = 0.88
        start_y = 0.20

        self.dataset_buttons = []
        for i in range(self.n_datasets):
            y_pos = start_y - i * (button_height + button_spacing)
            ax_btn = self.fig.add_axes([start_x, y_pos, button_width, button_height])
            btn = Button(ax_btn, self.dataset_names[i])
            btn.on_clicked(lambda event, dataset_idx=i: self._on_dataset_rescale_clicked(dataset_idx))
            self.dataset_buttons.append(btn)

        # Create reset button below dataset buttons
        reset_y = start_y - self.n_datasets * (button_height + button_spacing) - button_spacing
        self.ax_reset = self.fig.add_axes([start_x, reset_y, button_width, button_height])
        self.btn_reset = Button(self.ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset_clicked)

        # Connect right-click event handler to update slices
        self._click_cid = self.fig.canvas.mpl_connect('button_press_event', self._on_plot_click)

        # Initial plot
        self._update_plot()
        self.fig.subplots_adjust(bottom=0.30, hspace=0.3, wspace=0.3, top=0.92)

    def _create_sliders(self):
        """Create sliders for all axes except the one being plotted."""
        slider_height = 0.04
        base_slider_y = 0.20
        slider_count = 0

        for i in range(self.ndim):
            if i != self.current_x_axis:
                slider_y = base_slider_y - slider_count * (slider_height + 0.02)
                ax_slider = self.fig.add_axes([0.2, slider_y, 0.50, slider_height])
                self.slider_axes[i] = ax_slider
                max_val = len(self.values[i]) - 1
                slider = Slider(
                    ax_slider,
                    f'{self.labels[i]}',
                    0,
                    max_val,
                    valinit=self.slices.get(i, 0),
                    valstep=1
                )
                slider.on_changed(self._make_slider_callback(i))
                self.sliders[i] = slider
                slider_count += 1

    def _on_slider_changed(self, axis, value):
        """Handle slider changes."""
        self.slices[axis] = value
        # Save current zoom before updating plot
        self.zoom_xlim = self.ax_plot.get_xlim()
        self.zoom_ylim = self.ax_plot.get_ylim()
        self._update_plot()

    def _on_prev_axis_clicked(self, event):
        """Switch to previous axis."""
        self.current_x_axis = (self.current_x_axis - 1) % self.ndim
        self.zoom_xlim = None  # Reset zoom when changing axes
        self.zoom_ylim = None
        self._update_sliders()
        self._update_plot()

    def _on_next_axis_clicked(self, event):
        """Switch to next axis."""
        self.current_x_axis = (self.current_x_axis + 1) % self.ndim
        self.zoom_xlim = None  # Reset zoom when changing axes
        self.zoom_ylim = None
        self._update_sliders()
        self._update_plot()

    def _on_plot_click(self, event):
        """Handle right-click on the plot to update scroller positions."""
        if event.button != 3:  # Only handle right-click (button 3)
            return

        if event.xdata is None or event.ydata is None:
            return

        # Find all axes except the one being plotted
        other_axes = [i for i in range(self.ndim) if i != self.current_x_axis]

        if not other_axes:
            return

        # Update the first "other" axis based on click position
        first_other_axis = other_axes[0]

        # Find the closest index in the first other axis values
        axis_vals = self.values[first_other_axis]
        click_val = event.ydata
        closest_idx = int(np.argmin(np.abs(np.array(axis_vals) - click_val)))

        # Update the slice for this axis
        self.slices[first_other_axis] = closest_idx

        # Update slider if it exists
        if first_other_axis in self.sliders:
            self.sliders[first_other_axis].set_val(closest_idx)

        self._update_plot()

    def _on_dataset_rescale_clicked(self, dataset_idx):
        """Rescale plot to fit the currently displayed curve of a specific dataset."""
        # Reset zoom to default first
        self.zoom_xlim = None
        self.zoom_ylim = None

        # Get the slice data for this dataset
        slice_data = self._get_slice_data_1d(self.data_list[dataset_idx])

        # Calculate min and max for this dataset at current slices
        data_min = slice_data.min()
        data_max = slice_data.max()

        # Update y-axis range
        self.y_axis_min = data_min
        self.y_axis_max = data_max

        # Redraw plot with new range
        self._update_plot()

    def _on_reset_clicked(self, event):
        """Reset y-axis range to global min/max."""
        # Reset zoom to default first
        self.zoom_xlim = None
        self.zoom_ylim = None

        self.y_axis_min = self.global_data_min
        self.y_axis_max = self.global_data_max

        # Redraw plot with global range
        self._update_plot()

    def _get_slice_data_1d(self, arr):
        """Extract 1D slice from the array based on current axis and slices."""
        # Move the x_axis to position 0
        arr_moved = np.moveaxis(arr, self.current_x_axis, 0)

        # Build index tuple for slicing
        idx_tuple = [slice(None)]  # For x_axis (now at position 0)

        for i in range(self.ndim):
            if i == self.current_x_axis:
                continue
            idx_tuple.append(self.slices.get(i, 0))

        return arr_moved[tuple(idx_tuple)]

    def _update_plot(self):
        """Update the 1D plot."""
        x_vals = self.values[self.current_x_axis]

        # Get all data at current slices
        all_data = []
        for data_array in self.data_list:
            slice_data = self._get_slice_data_1d(data_array)
            all_data.append(slice_data)

        # Update line data
        for line, slice_data in zip(self.lines, all_data):
            line.set_data(x_vals, slice_data)

        # Restore zoom limits if they were saved (from slider changes)
        if self.zoom_xlim is not None and self.zoom_ylim is not None:
            self.ax_plot.set_xlim(self.zoom_xlim)
            self.ax_plot.set_ylim(self.zoom_ylim)
        else:
            # Update plot limits using stored y_axis range (doesn't change with sliders)
            self.ax_plot.set_xlim(np.array(x_vals).min(), np.array(x_vals).max())
            y_margin = (self.y_axis_max - self.y_axis_min) * 0.1 if self.y_axis_max > self.y_axis_min else 0.5
            self.ax_plot.set_ylim(self.y_axis_min - y_margin, self.y_axis_max + y_margin)

        # Update labels
        self.ax_plot.set_xlabel(self.labels[self.current_x_axis])
        self.ax_plot.set_ylabel('Value')

        # Build title showing all slice values
        title = f"1D plot: {self.labels[self.current_x_axis]}"
        slice_info = []
        for i in range(self.ndim):
            if i != self.current_x_axis:
                slice_val = self.values[i][self.slices.get(i, 0)]
                slice_info.append(f"{self.labels[i]}={slice_val:.3f}")
        if slice_info:
            title += " | " + ", ".join(slice_info)

        self.ax_plot.set_title(title)

        self.fig.canvas.draw_idle()


class MultiDimPlotter(PlotterBase):
    """Interactive plotter for multidimensional data arrays."""

    def __init__(self, data_list, labels=None, values=None, initial_xy_axes=(0, 1),
                 dataset_names=None, colors=None, linestyles=None, markers=None,
                 cmap_vmin=None, cmap_vmax=None):
        """
        Initialize the multidimensional data plotter.

        Parameters:
        -----------
        data_list : list of np.ndarray
            List of numpy arrays with identical shapes
        labels : list of str or None, optional
            Names of axes (e.g., ['X', 'Y', 'Z', 'Time']). If None, uses generic names (Axis 0, Axis 1, ...)
        values : list of list or None, optional
            Values for each axis (e.g., [[1,2,3], [4,5,6], ...]). If None, uses indices 0, 1, 2, ...
        initial_xy_axes : tuple of int, optional
            Initial axes indices for (x_axis, y_axis), default (0, 1)
        dataset_names : list of str, optional
            Names for each dataset. If None, uses 'Dataset 1', 'Dataset 2', ...
        colors : list of str, optional
            Colors for each line in 1D plot. If None, uses default colormap
        linestyles : list of str, optional
            Line styles for each dataset in 1D plot. If None, uses '-' (solid line)
        markers : list of str, optional
            Markers for each line in 1D plot. If None, uses no markers (empty string)
        cmap_vmin : float or None, optional
            Minimum value for 2D plot colormap. If None, uses data minimum
        cmap_vmax : float or None, optional
            Maximum value for 2D plot colormap. If None, uses data maximum
        """
        self.data_list = data_list
        self.n_datasets = len(data_list)

        # Handle labels
        self.ndim = data_list[0].ndim
        if labels is None:
            self.labels = [f'Axis {i}' for i in range(self.ndim)]
        else:
            self.labels = labels

        # Handle values
        if values is None:
            self.values = [list(range(s)) for s in data_list[0].shape]
        else:
            self.values = values

        self.shape = data_list[0].shape

        # Setup optional parameters using inherited method
        self._setup_optional_params(dataset_names, colors, linestyles, markers)

        # Store colormap options
        self.cmap_vmin = cmap_vmin
        self.cmap_vmax = cmap_vmax
        self.cbar = None  # Will store the colorbar object

        # Store 1D plot instance
        self.plot_1d = None

        # Store zoom limits for each subplot (for preserving zoom when using scrollers)
        self.zoom_limits = {}  # {ax_index: (xlim, ylim)}

        # Validate inputs (after handling None values)
        assert all(arr.shape == self.shape for arr in data_list), "All arrays must have same shape"
        assert len(self.labels) == self.ndim, f"Number of labels ({len(self.labels)}) must match dimensions ({self.ndim})"
        assert len(self.values) == self.ndim, f"Number of value lists ({len(self.values)}) must match dimensions ({self.ndim})"
        assert all(data_list[0].shape[i] == len(self.values[i]) for i in range(self.ndim))

        # Set initial axes
        self.x_axis, self.y_axis = initial_xy_axes
        self.slices = {}  # Store current slice indices for non-xy axes

        # Initialize slice indices for all axes except x and y
        for i in range(self.ndim):
            if i not in (self.x_axis, self.y_axis):
                self.slices[i] = 0

        # Create figure
        self.fig = plt.figure(figsize=(14, 8))

        # Calculate number of subplots needed
        # Arrange in grid: 1 row if <= 3 datasets, otherwise 2 rows
        if self.n_datasets <= 3:
            self.n_rows = 1
            self.n_cols = self.n_datasets
        else:
            self.n_cols = (self.n_datasets + 1) // 2
            self.n_rows = 2

        # Main plot area - create subplots for each dataset
        self.axes_main = []
        self.images = []
        for i in range(self.n_datasets):
            ax = self.fig.add_subplot(self.n_rows, self.n_cols, i + 1)
            self.axes_main.append(ax)
            self.images.append(None)

        # Connect right-click event handler ONCE to the canvas
        self._click_cid = self.fig.canvas.mpl_connect('button_press_event', self._make_click_callback())

        # Create sliders for non-xy axes
        self.sliders = {}
        self.slider_axes = {}  # Store axes for cleanup
        self._create_sliders()

        # Create axis swap buttons - positioned to the right of sliders with vertical spacing
        self.ax_swap_xy = self.fig.add_axes([0.78, 0.15, 0.06, 0.04])
        self.btn_swap_xy = Button(self.ax_swap_xy, 'Swap X↔Y')
        self.btn_swap_xy.on_clicked(self._on_swap_xy_clicked)

        # Create buttons to cycle through all axis combinations - with vertical spacing
        self.ax_prev_axes = self.fig.add_axes([0.78, 0.09, 0.06, 0.04])
        self.btn_prev_axes = Button(self.ax_prev_axes, '← Axes')
        self.btn_prev_axes.on_clicked(self._on_prev_axes_clicked)

        self.ax_next_axes = self.fig.add_axes([0.78, 0.03, 0.06, 0.04])
        self.btn_next_axes = Button(self.ax_next_axes, 'Axes →')
        self.btn_next_axes.on_clicked(self._on_next_axes_clicked)

        # Create colorbar sliders (vertical sliders to the right of colorbar)
        self.cbar_vmin_slider = None
        self.cbar_vmax_slider = None
        self.cbar_vmin_ax = None
        self.cbar_vmax_ax = None
        self.cbar_created = False  # Flag to track if colorbar has been created

        # Generate all axis pair combinations
        from itertools import combinations
        self.axis_combinations = list(combinations(range(self.ndim), 2))
        self.current_combo_idx = self.axis_combinations.index((self.x_axis, self.y_axis))

        # Adjust layout to give space for colorbar BEFORE first plot
        self.fig.subplots_adjust(bottom=0.3, hspace=0.3, wspace=0.35, top=0.92, right=0.92)

        # Initial plot
        self.update_plot()

    def _create_sliders(self):
        """Create sliders for non-xy axes."""
        slider_height = 0.04
        base_slider_y = 0.15
        slider_count = 0

        for i in range(self.ndim):
            if i not in (self.x_axis, self.y_axis):
                slider_y = base_slider_y - slider_count * (slider_height + 0.02)
                ax_slider = self.fig.add_axes([0.2, slider_y, 0.50, slider_height])
                self.slider_axes[i] = ax_slider
                max_val = self.shape[i] - 1
                slider = Slider(
                    ax_slider,
                    f'{self.labels[i]}',
                    0,
                    max_val,
                    valinit=self.slices.get(i, 0),
                    valstep=1
                )
                slider.on_changed(self._make_slider_callback(i))
                self.sliders[i] = slider
                slider_count += 1

    def _on_slider_changed(self, axis, value):
        """Handle slider changes."""
        self.slices[axis] = value
        # Save current zoom for all subplots before updating
        for idx, ax in enumerate(self.axes_main):
            self.zoom_limits[idx] = (ax.get_xlim(), ax.get_ylim())
        self.update_plot()

    def _make_click_callback(self):
        """Create a callback for click events with proper context."""
        def callback(event):
            self._on_plot_click(event)
        return callback

    def _on_swap_xy_clicked(self, event):
        """Swap X and Y axes."""
        self.x_axis, self.y_axis = self.y_axis, self.x_axis
        self.zoom_limits.clear()  # Reset zoom when changing axes
        # Find the normalized combination (min, max) in the list
        from itertools import combinations
        normalized_combo = tuple(sorted([self.x_axis, self.y_axis]))
        self.current_combo_idx = self.axis_combinations.index(normalized_combo)
        self.update_plot()

    def _on_prev_axes_clicked(self, event):
        """Switch to previous axis combination."""
        self.current_combo_idx = (self.current_combo_idx - 1) % len(self.axis_combinations)
        self.x_axis, self.y_axis = self.axis_combinations[self.current_combo_idx]
        self.zoom_limits.clear()  # Reset zoom when changing axes
        self._initialize_missing_slices()
        self._update_sliders()
        self.update_plot()

    def _on_next_axes_clicked(self, event):
        """Switch to next axis combination."""
        self.current_combo_idx = (self.current_combo_idx + 1) % len(self.axis_combinations)
        self.x_axis, self.y_axis = self.axis_combinations[self.current_combo_idx]
        self.zoom_limits.clear()  # Reset zoom when changing axes
        self._initialize_missing_slices()
        self._update_sliders()
        self.update_plot()

    def _initialize_missing_slices(self):
        """Initialize slice entries for any new axes that don't have them yet."""
        for i in range(self.ndim):
            if i not in (self.x_axis, self.y_axis) and i not in self.slices:
                self.slices[i] = 0

    def _update_sliders(self):
        """Hide/show sliders based on current x/y axes."""
        # Remove old sliders and their axes
        for i, slider in list(self.sliders.items()):
            try:
                self.fig.delaxes(slider.ax)
            except:
                pass

        self.sliders.clear()
        self.slider_axes.clear()

        # Recreate sliders for new non-xy axes
        self._create_sliders()

        self.fig.canvas.draw_idle()

    def _on_plot_click(self, event):
        # Check if right-click (button 3)
        if event.button != 3:
            return

        # Find which subplot was clicked
        clicked_ax = event.inaxes
        if clicked_ax is None or clicked_ax not in self.axes_main:
            return

        # Get the clicked coordinates
        x_click = event.xdata
        y_click = event.ydata

        if x_click is None or y_click is None:
            return

        # Find the nearest X index and Y index
        x_vals = self.values[self.x_axis]
        y_vals = self.values[self.y_axis]
        x_idx = int(np.argmin(np.abs(np.array(x_vals) - x_click)))
        y_idx = int(np.argmin(np.abs(np.array(y_vals) - y_click)))

        # If 1D plot exists, update it
        if self.plot_1d is not None:
            logger.debug("existing 1D plot")
            # Update all slice indices from the current 2D state
            for i in range(self.ndim):
                self.plot_1d.slices[i] = self.slices.get(i, 0)
            # Update the specific x and y indices from the click
            self.plot_1d.slices[self.x_axis] = x_idx
            self.plot_1d.slices[self.y_axis] = y_idx
            self.plot_1d.current_x_axis = self.x_axis

            # Update the slider values in the 1D plot
            for axis_idx, slider in self.plot_1d.sliders.items():
                new_val = self.plot_1d.slices.get(axis_idx, 0)
                slider.set_val(new_val)

            self.plot_1d._update_plot()

            # Bring 1D plot window to front
            try:
                # Try different methods depending on backend
                if hasattr(self.plot_1d.fig.canvas, 'manager') and self.plot_1d.fig.canvas.manager is not None:
                    # For Tkinter (TkAgg)
                    if hasattr(self.plot_1d.fig.canvas.manager, 'window'):
                        self.plot_1d.fig.canvas.manager.window.lift()
                        self.plot_1d.fig.canvas.manager.window.attributes('-topmost', True)
                        self.plot_1d.fig.canvas.manager.window.attributes('-topmost', False)
            except Exception as e:
                logger.warning("Could not raise window: %s", e)
        else:
            # Create new 1D plot with all datasets
            # Initialize slices with current 2D state, then override with click indices
            plot_1d_slices = self.slices.copy()
            plot_1d_slices[self.x_axis] = x_idx
            plot_1d_slices[self.y_axis] = y_idx

            self.plot_1d = Plot1D(
                data_list=self.data_list,
                x_axis=self.x_axis,
                y_axis=self.y_axis,
                labels=self.labels,
                values=self.values,
                slices=plot_1d_slices,
                initial_click_x_idx=x_idx,
                initial_click_y_idx=y_idx,
                dataset_names=self.dataset_names,
                colors=self.colors,
                linestyles=self.linestyles,
                markers=self.markers
            )
            # Closing the 1D window destroys its canvas but not this Python
            # object, so without this the "existing 1D plot" branch above
            # would keep trying to update/re-raise a dead window forever -
            # forget it here so the next right-click creates a fresh one.
            self.plot_1d.fig.canvas.mpl_connect('close_event', self._on_plot_1d_closed)
            plt.show(block=False)  # Display the window without blocking

    def _on_plot_1d_closed(self, event):
        """Reset the 1D plot reference once its window is closed."""
        self.plot_1d = None

    def _get_slice_data(self, arr):
        """Extract 2D slice from the array based on current axes and slices."""
        # Move axes to positions 0 and 1
        arr_moved = np.moveaxis(arr, [self.x_axis, self.y_axis], [0, 1])

        # Now apply the slices for other dimensions
        idx_tuple = [slice(None), slice(None)]
        for i in range(self.ndim):
            if i not in (self.x_axis, self.y_axis):
                idx_tuple.append(self.slices[i])

        return arr_moved[tuple(idx_tuple)]

    def update_plot(self):
        """Update the main plots with current data for each dataset."""
        # Extract 2D slices from all data arrays
        data_2d_list = [self._get_slice_data(arr) for arr in self.data_list]

        # Get axis values
        x_vals = self.values[self.x_axis]
        y_vals = self.values[self.y_axis]

        # Find global min and max across ALL data (not just current slice)
        global_min, global_max = self._get_global_min_max()

        # Use optional vmin/vmax if provided, otherwise use data limits
        vmin = self.cmap_vmin if self.cmap_vmin is not None else global_min
        vmax = self.cmap_vmax if self.cmap_vmax is not None else global_max

        # Calculate extent from center of first bin to center of last bin
        # Instead of from first value to last value
        x_vals_arr = np.array(x_vals)
        y_vals_arr = np.array(y_vals)

        # Calculate bin spacing
        x_spacing = (x_vals_arr[-1] - x_vals_arr[0]) / (len(x_vals_arr) - 1) if len(x_vals_arr) > 1 else 1
        y_spacing = (y_vals_arr[-1] - y_vals_arr[0]) / (len(y_vals_arr) - 1) if len(y_vals_arr) > 1 else 1

        # Extent from center of first bin to center of last bin
        x_min = x_vals_arr[0] - x_spacing / 2
        x_max = x_vals_arr[-1] + x_spacing / 2
        y_min = y_vals_arr[0] - y_spacing / 2
        y_max = y_vals_arr[-1] + y_spacing / 2

        extent = [x_min, x_max, y_min, y_max]

        for idx, (ax, data_2d) in enumerate(zip(self.axes_main, data_2d_list)):
            ax.clear()

            # Plot with consistent scaling
            im = ax.imshow(
                data_2d.T,
                extent=extent,
                aspect='auto',
                origin='lower',
                cmap='viridis',
                vmin=vmin,
                vmax=vmax
            )
            self.images[idx] = im

            ax.set_xlabel(self.labels[self.x_axis])
            ax.set_ylabel(self.labels[self.y_axis])

            # Set subplot title with dataset name
            if self.dataset_names and idx < len(self.dataset_names):
                ax.set_title(self.dataset_names[idx])
            else:
                ax.set_title(f'Dataset {idx + 1}')

            # Restore zoom limits if they were saved (from slider changes)
            if idx in self.zoom_limits:
                xlim, ylim = self.zoom_limits[idx]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        # Create colorbar only on first call (not on every update)
        if not self.cbar_created:
            # Create new colorbar (use the last image for colorbar)
            # Position it on the right side of the plot area
            self.cbar = self.fig.colorbar(self.images[-1], ax=self.axes_main, pad=0.08, shrink=0.8, fraction=0.046)
            self.cbar.set_label('Data Value')

            # Create vertical sliders for colorbar min/max at same height as buttons
            # Vertical slider for vmin (positioned at same height as buttons)
            self.cbar_vmin_ax = self.fig.add_axes([0.855, 0.04, 0.012, 0.16])
            self.cbar_vmin_slider = Slider(
                self.cbar_vmin_ax,
                'min',
                global_min,
                global_max,
                valinit=vmin,
                valstep=(global_max - global_min) / 100,
                orientation='vertical'
            )
            self.cbar_vmin_slider.on_changed(self._on_cbar_vmin_changed)
            # Hide numerical value display on slider
            self.cbar_vmin_slider.valtext.set_visible(False)

            # Vertical slider for vmax (positioned at same height as buttons, to the right)
            self.cbar_vmax_ax = self.fig.add_axes([0.880, 0.04, 0.012, 0.16])
            self.cbar_vmax_slider = Slider(
                self.cbar_vmax_ax,
                'max',
                global_min,
                global_max,
                valinit=vmax,
                valstep=(global_max - global_min) / 100,
                orientation='vertical'
            )
            self.cbar_vmax_slider.on_changed(self._on_cbar_vmax_changed)
            # Hide numerical value display on slider
            self.cbar_vmax_slider.valtext.set_visible(False)

            self.cbar_created = True
        else:
            # On subsequent updates, just update the image normalization
            for im in self.images:
                im.set_clim(vmin=vmin, vmax=vmax)
            self.cbar.update_normal(self.images[-1])

        # Update global title with slice information
        title = f"Data visualization"
        slice_info = []
        for i in range(self.ndim):
            if i not in (self.x_axis, self.y_axis):
                slice_val = self.values[i][self.slices[i]]
                slice_info.append(f"{self.labels[i]}={slice_val:.3f}")
        if slice_info:
            title += " (" + ", ".join(slice_info) + ")"

        self.fig.suptitle(title, fontsize=14)

        self.fig.canvas.draw_idle()

    def set_colormap_limits(self, vmin, vmax):
        """
        Dynamically change the colormap min and max values.

        Parameters:
        -----------
        vmin : float
            Minimum value for colormap
        vmax : float
            Maximum value for colormap
        """
        self.cmap_vmin = vmin
        self.cmap_vmax = vmax
        # Update all image objects with new limits
        for im in self.images:
            im.set_clim(vmin=vmin, vmax=vmax)
        # Update colorbar
        if self.cbar is not None:
            self.cbar.update_normal(self.images[-1])
        self.fig.canvas.draw_idle()

    def _on_cbar_vmin_changed(self, val):
        """Handle colorbar vmin slider changes."""
        if self.cbar_vmax_slider is not None and val > self.cbar_vmax_slider.val:
            self.cbar_vmin_slider.set_val(self.cbar_vmax_slider.val - 0.01)
            return
        self.set_colormap_limits(val, self.cbar_vmax_slider.val if self.cbar_vmax_slider else self.cmap_vmax)

    def _on_cbar_vmax_changed(self, val):
        """Handle colorbar vmax slider changes."""
        if self.cbar_vmin_slider is not None and val < self.cbar_vmin_slider.val:
            self.cbar_vmax_slider.set_val(self.cbar_vmin_slider.val + 0.01)
            return
        self.set_colormap_limits(self.cbar_vmin_slider.val if self.cbar_vmin_slider else self.cmap_vmin, val)
