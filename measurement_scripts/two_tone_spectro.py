import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import h5py as h5
import numpy as np
from tqdm.notebook import tqdm

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qcodes.dataset import Measurement
from qcodes.parameters import Parameter
from qcodes.station import Station

from legacy.instruments.AnaPicoAPUASYN20 import AnaPicoAPUASYN20  # noqa: E402
from stock_instruments.KeysightP5024A import KeysightP5024A  # noqa: E402
from stock_instruments.YokogawaGS200 import YokogawaGS200  # noqa: E402
from .qcodes_utils.measurement_run import (  # noqa: E402
    PeriodicFlush,
    instrument_metadata,
    open_experiment,
    safe_run,
)

from .base_measurement import BaseMeasurement


class TwoToneSpectro(BaseMeasurement):
    def __init__(
        self, save_path, circuit_path, vna_params, pico_params, dc_params, sweep_params
    ):
        """
        vna_params:dict(
            'visa_address':str,
            'nickname':str,
            config:dict)
        pico_params:dict(
            'visa_address':str,
            'nickname':str,
            'config':dict
            )
        dc_params:dict(
        'visa_address':str,
        'nickname':str,
        'config':dict
        )
        sweep_params:dict(
            'pico_freqs': shape (nfreqs,)
            'currents': shape (ncurrents,)
            'vna_freqs': shape (ncurrents,)
            'pico_powers': shape (nfreqs,)
        )
        """
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        vna_config = dict(vna_params_mod["config"])
        vna_config["sweep_type"] = "CW"
        vna_params_mod["config"] = vna_config
        self._vna_measurement_labels = tuple(
            label for label, _ in vna_config["measurements"]
        )
        if len(self._vna_measurement_labels) != 1:
            # The old script's `get_vna_data()` stacked every configured
            # measurement, but the fixed-shape `dset[i, j] = data`
            # assignment it wrote into only actually worked (via numpy
            # broadcasting) for exactly one - it silently only ever
            # supported a single measurement. Surface that rather than
            # reproduce it as silent data corruption.
            raise NotImplementedError(
                "TwoToneSpectro only supports a single vna measurement "
                f"(got {self._vna_measurement_labels!r}) - this matches the "
                "old script's actual (if undocumented) limitation."
            )

        # replaces pico_params['config'] outright, same as the old script
        pico_config = {
            "frequencies": (self.params["pico_freqs"][0],),
            "powers": (self.params["pico_powers"][0],),
        }

        self.vna = KeysightP5024A(
            name=vna_params_mod["nickname"],
            address=vna_params_mod["visa_address"],
            config=vna_config,
        )
        self.pico = AnaPicoAPUASYN20(
            name=pico_params["nickname"],
            address=pico_params["visa_address"],
            config=pico_config,
        )
        self.yoko = YokogawaGS200(
            name=dc_params["nickname"],
            address=dc_params["visa_address"],
            config=dc_params.get("config"),
        )
        self.instruments = (self.vna, self.pico, self.yoko)
        # call the super at the end!
        super().__init__(save_path, circuit_path)

    def execute(self):
        self._setup_sweep_coordinates()
        self._setup_instruments()
        self._setup_qcodes_experiment()
        self._run_sweep()

    # -- setup: sweep coordinates ---------------------------------------
    def _setup_sweep_coordinates(self):
        """Every array that defines this sweep. Pure math - no
        instrument, qcodes, or h5py calls."""
        self._currents = np.array(self.params["currents"])
        self._vna_freqs = np.array(self.params["vna_freqs"])
        self._pico_freqs = np.array(self.params["pico_freqs"])
        self._pico_powers = np.array(self.params["pico_powers"])

    # -- setup: instruments -----------------------------------------------
    def _setup_instruments(self):
        """Find out how many CW time-points the VNA is currently
        configured for, so the .h5 dataset can be pre-allocated with the
        right shape. Only talks to `self.vna`."""
        self._sweep_points = self.vna.points()

    def _setup_qcodes_experiment(self):
        """Open the sqlite .db/experiment this run gets added to, and
        snapshot the station + this sweep's params as run metadata. Only
        talks to the qcodes API."""
        self._experiment = open_experiment(
            self.save_file_path,
            experiment_name="two_tone_spectro",
            sample_name=Path(self.circuit_path).stem,
        )
        self._station = Station(self.vna, self.pico, self.yoko)
        self._run_metadata = instrument_metadata(
            self.instruments, self.params, self.circuit_path
        )

    def _setup_h5(self, file):
        """Pre-allocate the shared 3-D `data` array at its full final
        size, plus the four fixed 1-D axis datasets - exactly like the
        old script did - and a `time` array to hold every point's
        timestamp (the old script only ever kept the last one, see
        `_save_point_to_h5`). Only talks to h5py."""
        data_group = file["data"]
        data_group.create_dataset("Vna frequencies (Hz)", data=self._vna_freqs)
        data_group.create_dataset("Currents (A)", data=self._currents)
        data_group.create_dataset("Anapico frequencies (Hz)", data=self._pico_freqs)
        data_group.create_dataset("Anapico powers (dBm)", data=self._pico_powers)
        dset = data_group.create_dataset(
            "data",
            (len(self._currents), len(self._pico_freqs), self._sweep_points),
            dtype="complex128",
        )
        dset.dims[0].label = "Currents (A)"
        dset.dims[1].label = "Anapico frequency (Hz)"
        dset.dims[2].label = "CW Time (s)"
        time_dset = data_group.create_dataset(
            "time",
            (len(self._currents), len(self._pico_freqs)),
            dtype=h5.string_dtype(),
        )
        time_dset.dims[0].label = "Currents (A)"
        time_dset.dims[1].label = "Anapico frequency (Hz)"
        return dset, time_dset

    def _run_sweep(self):
        """Measure once per `(current, pico frequency)` pair, then hand
        the result to both the qcodes run and the legacy .h5. Interleaved
        on purpose (not "db first, .h5 exported afterwards"): this is
        the biggest sweep of the four (68017 points / ~69min in the
        original notebook) with no other natural checkpoint, so writing
        each point into both the open .h5 and the datasaver as soon as
        it's measured is what makes a crash/power-loss lose at most the
        last few points, never the whole file. `PeriodicFlush` only
        throttles how often that's pushed to disk with an OS-level
        flush, it never holds extra points in memory."""
        with safe_run(self.instruments), h5.File(self.save_file_path, "r+") as file:
            dset, time_dset = self._setup_h5(file)
            self.pico.output_enabled(True)

            meas = self._setup_qcodes_run()
            flush_gate = PeriodicFlush(interval=self.params.get("flush_interval", 300.0))

            bar = tqdm()
            bar.reset(total=len(self._currents) * len(self._pico_freqs))

            with meas.run() as datasaver:
                for key, value in self._run_metadata.items():
                    datasaver.dataset.add_metadata(key, value)

                for i_current, current in enumerate(self._currents):
                    self.yoko.current(current)
                    self.vna.cw(self._vna_freqs[i_current])
                    for i_pico_freq, pico_freq in enumerate(self._pico_freqs):

                        trace_data, time_str = self._measure_one_point(i_pico_freq)

                        self._save_point_to_h5(
                            dset, time_dset, i_current, i_pico_freq, time_str, trace_data
                        )

                        self._save_point_to_db(
                            datasaver, i_current, current, i_pico_freq, time_str, trace_data
                        )

                        bar.update()
                        if flush_gate.due():
                            file.flush()
                            datasaver.flush_data_to_database(block=False)

            bar.refresh()
            file.flush()

    def _measure_one_point(self, i_pico_freq):
        """Set the Anapico's frequency/power and read back the VNA
        trace. Only talks to `self.pico`/`self.vna` - no qcodes, no
        h5py."""
        pico_freq = self._pico_freqs[i_pico_freq]
        self.pico.frequency(pico_freq)
        self.pico.power(self._pico_powers[i_pico_freq])
        self.vna.run_averaging()
        time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        (vna_meas,) = self._vna_measurement_labels
        trace_data = self.vna.read_raw_data(vna_meas)
        return trace_data, time_str

    def _setup_qcodes_run(self):
        """Register the Measurement + Parameters for this sweep. Sets
        `self._qc_*`, read by `_save_point_to_db`."""
        (vna_meas,) = self._vna_measurement_labels
        self._qc_current_param = Parameter(
            "current", label="DC current", unit="A", get_cmd=None, set_cmd=None
        )
        self._qc_pico_freq_param = Parameter(
            "pico_frequency",
            label="Anapico frequency",
            unit="Hz",
            get_cmd=None,
            set_cmd=None,
        )
        self._qc_vna_freq_setting_param = Parameter(
            "vna_frequency_setting",
            label="VNA CW frequency",
            unit="Hz",
            get_cmd=None,
            set_cmd=None,
        )
        self._qc_pico_power_param = Parameter(
            "pico_power", label="Anapico power", unit="dBm", get_cmd=None, set_cmd=None
        )
        self._qc_time_param = Parameter("time", label="Timestamp", get_cmd=None, set_cmd=None)
        self._qc_trace_param = Parameter(
            vna_meas, label=vna_meas, unit="", get_cmd=None, set_cmd=None
        )

        meas = Measurement(name="two_tone_spectro", exp=self._experiment, station=self._station)
        meas.register_parameter(self._qc_current_param, paramtype="numeric")
        meas.register_parameter(self._qc_pico_freq_param, paramtype="numeric")
        meas.register_parameter(
            self._qc_vna_freq_setting_param,
            setpoints=(self._qc_current_param,),
            paramtype="numeric",
        )
        meas.register_parameter(
            self._qc_pico_power_param,
            setpoints=(self._qc_pico_freq_param,),
            paramtype="numeric",
        )
        meas.register_parameter(
            self._qc_time_param,
            setpoints=(self._qc_current_param, self._qc_pico_freq_param),
            paramtype="text",
        )
        meas.register_parameter(
            self._qc_trace_param,
            setpoints=(self._qc_current_param, self._qc_pico_freq_param),
            paramtype="array",
        )
        return meas

    def _save_point_to_db(self, datasaver, i_current, current, i_pico_freq, time_str, trace_data):
        """One point -> one row in the qcodes run."""
        datasaver.add_result(
            (self._qc_current_param, current),
            (self._qc_pico_freq_param, self._pico_freqs[i_pico_freq]),
            (self._qc_vna_freq_setting_param, self._vna_freqs[i_current]),
            (self._qc_pico_power_param, self._pico_powers[i_pico_freq]),
            (self._qc_time_param, time_str),
            (self._qc_trace_param, trace_data),
        )

    def _save_point_to_h5(self, dset, time_dset, i_current, i_pico_freq, time_str, trace_data):
        """One point -> one cell of the pre-allocated `data` array, and
        the matching cell of `time`. The old script overwrote one shared
        dataset-level attr on every one of the (n_current*n_pico_freq)
        writes, so only the very last timestamp ever survived - fixed
        here, every point's timestamp is now kept."""
        dset[i_current, i_pico_freq] = trace_data
        time_dset[i_current, i_pico_freq] = time_str
