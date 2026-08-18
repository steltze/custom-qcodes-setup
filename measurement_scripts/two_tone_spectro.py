import numpy as np
import h5py as h5
from datetime import datetime
from copy import deepcopy
from alive_progress import alive_bar

from .base_measurement import BaseMeasurement
from ..instruments.vna_P5024A import VNA_P5024A
from ..instruments.anapico import Anapico1
from ..instruments.Yokogs200 import Yokogs200

class TwoToneSpectro(BaseMeasurement):
    def __init__(self, save_path, circuit_path,
                 vna_params, pico_params, dc_params, sweep_params):
        '''
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
        '''
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        vna_params_mod['config']['sweep_type'] = 'CW'
        pico_params_mod = deepcopy(pico_params)
        pico_params_mod['config'] = {'frequencies':(self.params['pico_freqs'][0],), 'powers':(self.params['pico_powers'][0],)}
        
        self.vna = VNA_P5024A(**vna_params_mod)
        self.pico = Anapico1(**pico_params_mod)
        self.yoko = Yokogs200(**dc_params) #default config
        self.instruments = (self.vna, self.pico, self.yoko)
        #call the super at the end!
        super().__init__(save_path, circuit_path)
        
    def get_vna_data(self):
        return np.array(
            [self.vna.channel.read_raw_data(meas_label) for (meas_label, _) in self.vna.config['measurements']]
            )
    
    def execute(self):
        currents = np.array(self.params['currents'])
        vna_freqs = np.array(self.params['vna_freqs'])
        pico_freqs = np.array(self.params['pico_freqs'])
        pico_powers = np.array(self.params['pico_powers'])
        with h5.File(self.save_file_path, "r+") as file:
            data_group = file['data']
            data_group.create_dataset("Vna frequencies (Hz)", data=vna_freqs)
            data_group.create_dataset("Currents (A)", data=currents)
            data_group.create_dataset("Anapico frequencies (Hz)", data=pico_freqs)
            data_group.create_dataset("Anapico powers (dBm)", data=pico_powers)
        self.yoko.reset_config
        loop_number = 0
        total_loops = len(currents)*len(pico_freqs)
        number_length = len(str(total_loops))
        # prepare dataset
        with h5.File(self.save_file_path, 'r+') as file:
            data_group = file['data']
            dset = data_group.create_dataset(
                "data",
                (len(currents), len(pico_freqs), self.vna.channel.sweep_points),
                dtype='complex128')
            dset.dims[0].label = "Currents (A)"
            dset.dims[1].label = "Anapico frequency (Hz)"
            #dset.dims[2].label = "Sig1Sig2/Sig2Sig1"
            dset.dims[2].label = "CW Time (s)"
        # turn on anapico
        self.pico.channels[0].output = "ON"
        # loop
        with alive_bar(total_loops, force_tty=True, spinner="triangles", max_cols=150) as bar:
            for i_current, current in enumerate(currents):
                self.yoko.current = current
                self.vna.channel.frequency_cw = vna_freqs[i_current]
                for i_pico_freq, pico_freq in enumerate(pico_freqs):
                    loop_number += 1
                    self.pico.channels[0].frequency = pico_freq
                    self.pico.channels[0].power = pico_powers[i_pico_freq]
                    now = datetime.now()
                    time_str = now.strftime("%m/%d/%Y, %H:%M:%S")
                    self.vna.channel.run_averaging()
                    if loop_number==1:
                        self.vna.channel.autoscale_all(1)
                    data = self.get_vna_data()
                    with h5.File(self.save_file_path, "r+") as file:
                        dset = file['data']['data']
                        dset.attrs['time'] = time_str
                        dset[i_current, i_pico_freq] = data
                    bar()
        self.pico.channels[0].output = "OFF"
        self.yoko.reset_config
            
            
        
        
        