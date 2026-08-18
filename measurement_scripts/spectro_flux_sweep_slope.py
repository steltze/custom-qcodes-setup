import numpy as np
import h5py as h5
from datetime import datetime
from copy import deepcopy
from alive_progress import alive_bar

from .base_measurement import BaseMeasurement
from ..instruments.vna_P5024A import VNA_P5024A
from ..instruments.Yokogs200 import Yokogs200

class SpectroDCSweepSlope(BaseMeasurement):
    def __init__(self, save_path, circuit_path,
                 vna_params, dc_params, sweep_params):
        '''
        vna_params:dict('visa_address':str, 'nickname':str, config:dict)
            'measurements' will be overwritten.
        dc_params:dict(
            'visa_address':str,
            'nickname':str,
            'config':dict
            )
        'sweep_params':dict(
                'channel':int, 
                'freq_start':float,
                'freq_end':float,
                'n_freq':int,
                'power_start':float,
                'power_end':float,
                'n_power':int,
                'pow_slope':float)
            )
        '''
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        if not('config' in vna_params_mod.keys()):
            vna_params_mod['config'] = {'measurements':None}
        ## Antho's device mapping:
        if ('line' in vna_params_mod['config'].keys()) and vna_params_mod['config']['line'] == 'pump':
            vna_params_mod['config']['measurements'] = (
                ('Sig1Sig1', 'S43'),
                ('Sig1Sig2', 'S41')    
            )
        else:
            vna_params_mod['config']['measurements'] = (
                ('Sig1Sig1', 'S41'),
                ('Sig2Sig1', 'S21'),
                ('Sig1Sig2', 'S43'),
                ('Sig2Sig2', 'S23')       
            )
        
        self.vna = VNA_P5024A(**vna_params_mod)
        self.yoko = Yokogs200(**dc_params) #default config
        ## ?? self.pump_channel = self.pump.channels[self.params['channel']-1] ??
        self.instruments = (self.vna, self.yoko)
        #call the super at the end!
        super().__init__(save_path, circuit_path)
        
    def execute(self):
        pow_slope = self.params['power_slope']
        self.vna.channel.power_slope = pow_slope
        self.vna.channel.power_slope_state = 1
        dc_current = np.linspace(
            self.params['current_start'],
            self.params['current_end'],
            self.params['n_current']
            )
        vna_meas = [name for (name,_) in self.vna.config['measurements']]
        freq_data = self.vna.channel.read_freq_data(meas_name=vna_meas[0])
        with h5.File(self.save_file_path, "r+") as file:
            data_group = file['data']
            data_group.create_dataset("vna frequencies", data=freq_data)
        #turn on pump, safe value
        self.yoko.reset_config
        loop_number = 0
        total_loops = len(dc_current)
        number_length = len(str(total_loops))
        with alive_bar(total_loops, force_tty=True, spinner="triangles", max_cols=150) as bar:
            for current in dc_current:
                self.yoko.current = current
                loop_number += 1
                now = datetime.now()
                time_str = now.strftime("%m/%d/%Y, %H:%M:%S")
                self.vna.channel.run_averaging()

                self.vna.channel.autoscale_all(1)

                with h5.File(self.save_file_path, "r+") as file:
                    data_group = file['data']
                    data = np.array([self.vna.channel.read_raw_data(meas) for meas in vna_meas])
                    dset = data_group.create_dataset(str(loop_number).zfill(number_length),
                        data=data.transpose())
                    dset.attrs['time'] = time_str
                    dset.attrs['Current (mA)'] = current*1e-3
                    for i_meas, meas in enumerate(vna_meas):
                        dset.attrs[f'Data col {i_meas}'] = meas
                bar()
        self.yoko.reset_config
        # self.yoko.close_connection