from read_data_utils import *

filename = r'\awg_pump_20260820_101822'

data_path = r""

data = load_fs_fp_map_data(filename, data_path, save=False)
data = np.einsum('cpfaij->cpfaji', data)
print(data.shape)
"""
# correction
data = np.einsum('cpfaij->cpafij', data)
data = correct_from_paths_T10_T23(data, T10_path, T23_path)
data = np.einsum('cpafij->cpfaij', data)
"""
f = h5py.File(data_path + filename + ".h5", 'r')
measurement_metadata = f['metadata']['measurement']
pump_freqs = np.array(measurement_metadata['freqs'])
amps_array = np.linspace(np.array(measurement_metadata['main_amp_starts']),
                         np.array(measurement_metadata['main_amp_ends']),
                         np.array(measurement_metadata['n_main_amp'], dtype=int))  # shape (amp, pump freq)
amps_array = amps_array  #/2  # peak to peak awg
amps = amps_array[:, 0]
freq_spec = np.array(f['metadata']['instruments']['vna_awgpump_bringup']['config']['freq_spec'])
signal_freqs = np.linspace(freq_spec[0], freq_spec[1], int(freq_spec[2]))
currents = np.linspace(np.array(measurement_metadata['current_start']), np.array(measurement_metadata['current_end']), np.array(measurement_metadata['n_current'], dtype=int))[:data.shape[0]]


MultiDimPlotter([20*np.log10(np.abs(data[..., 0, 1])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 0, 1], axis=3))), 20*np.log10(np.abs(data[..., 1, 0])) - 20*np.log10(np.abs(np.expand_dims(data[..., 0, 1, 0], axis=3)))], labels=['Current (mA)', 'Pump Freq (GHz)', 'Signal Freq (GHz)', 'Pump Amp (V)'], values=[currents*1e3, pump_freqs*1e-9, signal_freqs*1e-9, amps], dataset_names=['|S12| dB', '|S21| dB'], initial_xy_axes=(1, 2), cmap_vmin=-10, cmap_vmax=5)
plt.show()