import os
import h5py as h5

class BaseMeasurement():
    def __init__(self, save_file_path, circuit_path):
        '''
        To call AFTER the initialization of the specific measurement.
        Before that, put all instruments used in self.instruments.
        This is for proper saving purposes.
        '''
        self.meas_type = type(self)     
        self.save_file_path = save_file_path
        self.circuit_path = circuit_path
        if not(os.path.isfile(self.circuit_path)):
            raise FileNotFoundError(f"Circuit file not found at {self.circuit_path}. \
                                Will not proceed, save your circuit correctly.")
        if not(hasattr(self, "instruments")):
            raise AttributeError("Measurement has no attribute 'instruments'.\
                                 Will not proceed, add used instruments to \
                                 self.instruments in Measurement code.")
        if not(hasattr(self, "params")):
            raise AttributeError("Measurement has no attribute 'params'.\
                                 Will not proceed, measurement parameters \
                                 should be an attribute.")
        self.save_measurement_info()
        
    def save_measurement_info(self):
        if os.path.isfile(self.save_file_path):
            raise FileExistsError(f"File {self.save_file_path} exists. \
                                  Will not overwrite, change measurement name \
                                  and save path.")
        with h5.File(self.save_file_path, "w") as file:
            meta = file.create_group("metadata")
            instruments = meta.create_group("instruments")
            for instru in self.instruments:
                instru_info = instru.get_config_info()
                instru_group = instruments.create_group(instru_info['nickname'])
                instru_group.create_dataset('type', data=str(type(instru)))
                instru_group.create_dataset('visa_address', data=instru_info['visa_address'])
                config_group = instru_group.create_group('config')
                for key in instru_info['config'].keys():
                    config_group.create_dataset(key, data=instru_info['config'][key])
            meas_group = meta.create_group("measurement")
            meas_group.create_dataset("meas_type", data=str(self.meas_type))
            for key in self.params.keys():
                meas_group.create_dataset(key, data=self.params[key])
            circuit_group = meta.create_group("circuit")
            circuit_group.create_dataset("circuit_path", data=self.circuit_path)
            file.create_group("data")