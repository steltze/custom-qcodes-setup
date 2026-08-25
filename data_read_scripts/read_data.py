# from read_data_utils import *

# filename = r'\awg_pump_20260820_165226'
# data_path = r"C:\Users\REDACTED\Documents\Codes\qcodes_migration\custom-qcodes-setup\data\bringup"

# data = load_fs_fp_map_data(filename, data_path, save=False)
# print(data.shape)
# data = np.einsum('cpfaij->cpfaji', data)
# """
# # correction
# data = np.einsum('cpfaij->cpafij', data)
# data = correct_from_paths_T10_T23(data, T10_path, T23_path)
# data = np.einsum('cpafij->cpfaij', data)
# """
# f = h5py.File(data_path + filename + ".h5", 'r')
# measurement_metadata = f['metadata']['measurement']
# pump_freqs = np.array(measurement_metadata['freqs'])
# amps_array = np.linspace(np.array(measurement_metadata['main_amp_starts']),
#                          np.array(measurement_metadata['main_amp_ends']),
#                          np.array(measurement_metadata['n_main_amp'], dtype=int))  # shape (amp, pump freq)
# amps_array = amps_array  #/2  # peak to peak awg
# amps = amps_array[:, 0]
# freq_spec = np.array(f['metadata']['instruments']['platoVNA']['config']['freq_spec'])
# signal_freqs = np.linspace(freq_spec[0], freq_spec[1], int(freq_spec[2]))
# currents = np.linspace(np.array(measurement_metadata['current_start']), np.array(measurement_metadata['current_end']), np.array(measurement_metadata['n_current'], dtype=int))[:data.shape[0]]

# import matplotlib as mpl
# import h5py
# import pickle as pkl
# # mpl.use('TkAgg')
# MultiDimPlotter([20*np.log10(np.abs(data[..., 0, 1])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 0, 1], axis=3))), 20*np.log10(np.abs(data[..., 1, 0])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 1, 0], axis=3)))], labels=['Current (mA)', 'Pump Freq (GHz)', 'Signal Freq (GHz)', 'Pump Amp (V)'], values=[currents*1e3, pump_freqs*1e-9, signal_freqs*1e-9, amps], dataset_names=['|S12| dB', '|S21| dB'], initial_xy_axes=(1, 2), cmap_vmin=-10, cmap_vmax=5)
# plt.show()
# MultiDimPlotter([20*np.log10(np.abs(data[..., 0, 1])), 20*np.log10(np.abs(data[..., 1, 0]))], labels=['Current (mA)', 'Pump Freq (GHz)', 'Signal Freq (GHz)', 'Pump Amp (V)'], values=[currents*1e3, pump_freqs*1e-9, signal_freqs*1e-9, amps], dataset_names=['|S12| dB', '|S21| dB'], initial_xy_axes=(1, 2), cmap_vmin=-10, cmap_vmax=5)
# plt.show()

from read_data_utils import *

filename = r'awg_pump_20260825_113258'

data_path = r"/home/REDACTED/Stelios/PPM/Quantic/Qcodes_Migration/custom-qcodes-setup/data/"

h5_reader = H5DataReader(data_path)

data = h5_reader.load_fs_fp_map(filename, save=False)
data = np.einsum('cpfaij->cpfaji', data)
print(data.shape)
"""
# correction
data = np.einsum('cpfaij->cpafij', data)
data = correct_from_paths_T10_T23(data, T10_path, T23_path)
data = np.einsum('cpafij->cpfaij', data)
"""
metadata = h5_reader.read_metadata(filename + ".h5")
measurement_metadata = metadata['measurement']
pump_freqs = np.array(measurement_metadata['freqs'])
amps_array = np.linspace(np.array(measurement_metadata['main_amp_starts']),
                         np.array(measurement_metadata['main_amp_ends']),
                         np.array(measurement_metadata['n_main_amp'], dtype=int))  # shape (amp, pump freq)
amps_array = amps_array  #/2  # peak to peak awg
amps = amps_array[:, 0]
freq_spec = np.array(metadata['instruments']['platoVNA']['config']['freq_spec'])
signal_freqs = np.linspace(freq_spec[0], freq_spec[1], int(freq_spec[2]))
currents = np.linspace(np.array(measurement_metadata['current_start']), np.array(measurement_metadata['current_end']), np.array(measurement_metadata['n_current'], dtype=int))[:data.shape[0]]


# MultiDimPlotter([20*np.log10(np.abs(data[..., 0, 1])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 0, 1], axis=3))), 20*np.log10(np.abs(data[..., 1, 0])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 1, 0], axis=3)))], labels=['Current (mA)', 'Pump Freq (GHz)', 'Signal Freq (GHz)', 'Pump Amp (V)'], values=[currents*1e3, pump_freqs*1e-9, signal_freqs*1e-9, amps], dataset_names=['|S12| dB', '|S21| dB'], initial_xy_axes=(1, 2), cmap_vmin=-10, cmap_vmax=5)
# plt.show()

MultiDimPlotter([20*np.log10(np.abs(data[..., 0, 1])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 0, 1], axis=3))), 20*np.log10(np.abs(data[..., 1, 0])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 1, 0], axis=3)))], labels=['Current (mA)', 'Pump Freq (GHz)', 'Signal Freq (GHz)', 'Pump Amp (V)'], values=[currents*1e3, pump_freqs*1e-9, signal_freqs*1e-9, amps], dataset_names=['|S12| dB', '|S21| dB'], initial_xy_axes=(1, 2), cmap_vmin=-10, cmap_vmax=5)
plt.show()
# MultiDimPlotter([20*np.log10(np.abs(data[..., 0, 1])), 20*np.log10(np.abs(data[..., 1, 0]))], labels=['Current (mA)', 'Pump Freq (GHz)', 'Signal Freq (GHz)', 'Pump Amp (V)'], values=[currents*1e3, pump_freqs*1e-9, signal_freqs*1e-9, amps], dataset_names=['|S12| dB', '|S21| dB'], initial_xy_axes=(1, 2), cmap_vmin=-10, cmap_vmax=5)
# plt.show()


import json

# ---------------------------------------------------------------------------
# Example: reading the same run from the QCoDeS database instead of .h5
# ---------------------------------------------------------------------------
# Every measurement_scripts/*.py script also writes each run into one shared
# experiments.db next to the .h5 files (see
# measurement_scripts/qcodes_utils/measurement_run.py::open_experiment).
db_reader = QCodesDatabaseReader(Path(data_path) / "experiments.db")

for run in db_reader.list_runs():
    print(run)

# "vna_calib_slope_custom_meas" - one independent-parameter tree (frequency),
# so it's the one experiment in this repo `load_run_dataframe` handles
# directly; see that method's docstring for the multi-instrument-sweep case.
df = db_reader.load_run_dataframe(experiment_name="vna_calib_slope_custom_meas")
print(df.head())

# A multi-instrument sweep (e.g. the AWG-pump run analysed above) has more
# than one independent-parameter shape, so to_pandas_dataframe() raises -
# get_parameter_data() always works instead:
awg_pump_run = db_reader.load_latest_run("spectro_awgPump_sweep_flux")
pdata = awg_pump_run.get_parameter_data()
print({dependent: {name: arr.shape for name, arr in cols.items()} for dependent, cols in pdata.items()})

# ---------------------------------------------------------------------------
# Rebuild the *whole* sweep from the database and render it the same way as
# the .h5 section above (2 background-subtracted cross-term traces, sliders
# for Current and Pump Amp) - not just the one (current, pump_freq) point
# `load_latest_run` above returns. `_run_sweep`
# (measurement_scripts/spectro_awgPump_sweep_variable_ranges_simpNOCompOnly_powSlope_sweep_flux.py)
# writes the *whole* sweep into a single qcodes run - `meas.run()` is only
# called once, outside the current/pump_freq/main_amp loops, and current,
# pump_freq and main_amp are all registered as independent (setpoint)
# parameters on that one run - so `get_parameter_data()` on it already holds
# every point; `measurement_params` (the sweep_params this class was
# constructed with) just tells us the grid shape to reshape it into.
awg_pump_run_id = 11
full_sweep_run = db_reader.load_run(awg_pump_run_id)
sweep_params = json.loads(full_sweep_run.metadata["measurement_params"])

def string_to_array(s):
    return np.fromstring(s.strip("[]"), sep=" ")

db_pump_freqs = string_to_array(sweep_params["freqs"])
db_n_freq = len(db_pump_freqs)
db_n_current = sweep_params["n_current"]
db_n_main_amp = sweep_params["n_main_amp"]
db_currents = np.linspace(sweep_params["current_start"], sweep_params["current_end"], db_n_current)
db_amps_array = np.linspace(
    string_to_array(sweep_params["main_amp_starts"]), string_to_array(sweep_params["main_amp_ends"]), db_n_main_amp
)  # shape (amp, pump freq)
db_amps = db_amps_array[:, 0]

# `_run_sweep` adds one row per point, in row-major (current outer,
# pump_freq, main_amp inner) order - so the raw (n_points, n_vna_freq)
# array reshapes directly into (current, pump_freq, main_amp, signal_freq);
# no transpose needed until after that reshape. A run a crash/Ctrl-C cut
# short (safe_run still leaves a valid, partially-filled .db - see
# measurement_run.py) has fewer rows than the full planned grid, so pad the
# missing trailing points with NaN instead of assuming completion;
# MultiDimPlotter's _get_global_min_max already skips NaNs.
grid_pdata = full_sweep_run.get_parameter_data()
db_signal_freqs = grid_pdata["Sig1Sig2"]["vna_frequency"][0, :]
n_vna_freq = len(db_signal_freqs)
grid_shape = (db_n_current, db_n_freq, db_n_main_amp, n_vna_freq)
n_expected = db_n_current * db_n_freq * db_n_main_amp

def to_grid(raw):
    n_captured = raw.shape[0]
    if n_captured < n_expected:
        print(f"run {awg_pump_run_id}: {n_captured}/{n_expected} points captured "
              f"(sweep interrupted) - padding the rest with NaN")
        pad = np.full((n_expected - n_captured, n_vna_freq), np.nan, dtype=raw.dtype)
        raw = np.concatenate([raw, pad], axis=0)
    elif n_captured > n_expected:
        raise ValueError(
            f"run {awg_pump_run_id} has {n_captured} points, more than the {n_expected} "
            f"its own measurement_params grid ({db_n_current}x{db_n_freq}x{db_n_main_amp}) implies"
        )
    return raw.reshape(grid_shape)

s12 = to_grid(grid_pdata["Sig1Sig2"]["Sig1Sig2"])
s21 = to_grid(grid_pdata["Sig2Sig1"]["Sig2Sig1"])
# swap the last two axes to match the .h5 section's
# (current, pump_freq, signal_freq, pump_amp) order.
s12 = np.moveaxis(s12, -1, -2)
s21 = np.moveaxis(s21, -1, -2)

# same background subtraction as the .h5 section: dB relative to each
# point's own value at pump_amp index 0.
s12_db = 20 * np.log10(np.abs(s12)) - 20 * np.log10(np.abs(s12[..., :1]))
s21_db = 20 * np.log10(np.abs(s21)) - 20 * np.log10(np.abs(s21[..., :1]))

MultiDimPlotter(
    [s12_db, s21_db],
    labels=["Current (mA)", "Pump Freq (GHz)", "Signal Freq (GHz)", "Pump Amp (V)"],
    values=[db_currents * 1e3, db_pump_freqs * 1e-9, db_signal_freqs * 1e-9, db_amps],
    dataset_names=["|S12| dB", "|S21| dB"],
    initial_xy_axes=(1, 2),
    cmap_vmin=-10,
    cmap_vmax=5,
)
plt.show()
