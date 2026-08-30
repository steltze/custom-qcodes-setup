import logging

from read_data_utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

filename = r'26_08_26_map_fp_fs_left'

data_path = r""

h5_reader = H5DataReader(data_path)

data = h5_reader.load_fs_fp_map(filename, save=False)
data = np.einsum('cpfaij->cpfaji', data)
logger.info("data shape: %s", data.shape)
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

db_reader = QCodesDatabaseReader(Path(data_path) / "experiments.db")

for run in db_reader.list_runs():
    logger.info("%s", run)

df = db_reader.load_run_dataframe(experiment_name="vna_calib_slope_custom_meas")
logger.info("%s", df.head())

awg_pump_run = db_reader.load_latest_run("spectro_awgPump_sweep_flux")
pdata = awg_pump_run.get_parameter_data()
logger.info(
    "%s", {dependent: {name: arr.shape for name, arr in cols.items()} for dependent, cols in pdata.items()}
)

awg_pump_run_id = 12
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

grid_pdata = full_sweep_run.get_parameter_data()
db_signal_freqs = grid_pdata["Sig1Sig2"]["vna_frequency"][0, :]
n_vna_freq = len(db_signal_freqs)
grid_shape = (db_n_current, db_n_freq, db_n_main_amp, n_vna_freq)
n_expected = db_n_current * db_n_freq * db_n_main_amp

def to_grid(raw):
    n_captured = raw.shape[0]
    if n_captured < n_expected:
        logger.warning(
            "run %s: %d/%d points captured (sweep interrupted) - padding the rest with NaN",
            awg_pump_run_id, n_captured, n_expected,
        )
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

s12 = np.moveaxis(s12, -1, -2)
s21 = np.moveaxis(s21, -1, -2)

s12_db = 20 * np.log10(np.abs(s12)) - 20 * np.log10(np.abs(s12[..., :1]))
s21_db = 20 * np.log10(np.abs(s21)) - 20 * np.log10(np.abs(s21[..., :1]))

MultiDimPlotter(
    [s21_db, s12_db],
    labels=["Current (mA)", "Pump Freq (GHz)", "Signal Freq (GHz)", "Pump Amp (V)"],
    values=[db_currents * 1e3, db_pump_freqs * 1e-9, db_signal_freqs * 1e-9, db_amps],
    dataset_names=["|S12| dB", "|S21| dB"],
    initial_xy_axes=(1, 2),
    cmap_vmin=-10,
    cmap_vmax=5,
)
plt.show()
