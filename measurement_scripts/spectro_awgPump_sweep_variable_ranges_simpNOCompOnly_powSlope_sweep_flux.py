import os
import numpy as np
import h5py as h5
from datetime import datetime
from copy import deepcopy
from tqdm.notebook import tqdm

from time import sleep

from .base_measurement import BaseMeasurement
from ..instruments.awg_M8195A import AWG_M8195A
from ..instruments.vna_P5024A import VNA_P5024A
from ..instruments.Yokogs200 import Yokogs200

class SpectroAWGPumpSweepFIRSimpNOCompSweepFlux(BaseMeasurement):
    def __init__(self, save_path, circuit_path,
                 vna_params, awg_params, dc_params, sweep_params):
        '''
        vna_params:dict('visa_address':str, 'nickname':str, config:dict)
            'measurements' will be overwritten.
        awg_params:dict(
            'visa_address':str,
            'nickname':str,
            'config':dict
            )
        'sweep_params':dict(
                'main_channel':str, 
                'freqs': list(float) of len n_freq (def of n_freq)
                'main_amp_starts':list(float) of len n_freq
                'main_amp_ends':list(float) of len n_freq,
                'n_main_amp':int,
                'power_slope':float
                )
            )
        '''
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        if not('config' in vna_params_mod.keys()):
            vna_params_mod['config'] = {'measurements':None}
        vna_params_mod['config']['measurements'] = (
            ('Sig1Sig1', 'S23'),
            ('Sig2Sig1', 'S43'),
            ('Sig1Sig2', 'S21'),
            ('Sig2Sig2', 'S41')       
        )
        self.vna = VNA_P5024A(**vna_params_mod)
        self.pump = AWG_M8195A(**awg_params)
        self.yoko = Yokogs200(**dc_params) #default config
        self.instruments = (self.vna, self.pump, self.yoko)
        self.main_ch = self.params['main_channel']
        #call the super at the end!
        super().__init__(save_path, circuit_path)
        
    def execute(self):
        dc_current = np.linspace(
            self.params['current_start'],
            self.params['current_end'],
            self.params['n_current']
            )
        pow_slope = self.params['power_slope']
        self.vna.channel.power_slope = pow_slope
        self.vna.channel.power_slope_state = 1
        pump_freqs = self.params['freqs']
        n_freq = len(pump_freqs)
        
        main_pump_amps = np.zeros((n_freq, self.params['n_main_amp']))
        for i in range(n_freq):
            main_pump_amps[i] = np.linspace(
                self.params['main_amp_starts'][i],
                self.params['main_amp_ends'][i],
                self.params['n_main_amp']
                )
        self.vna.channel.run_averaging()
        freq_data = self.vna.channel.read_freq_data()
        with h5.File(self.save_file_path, "r+") as file:
            data_group = file['data']
            data_group.create_dataset("Vna frequencies (Hz)", data=freq_data)
            data_group.create_dataset("Pump frequencies (Hz)", data=pump_freqs)
            data_group.create_dataset("Main pump amps (Vpk-pk)", data=main_pump_amps)
            data_group.create_dataset("DC current (A)", data=dc_current)
        #prepare pump, safe value
        self.yoko.reset_config
        self.pump.configure(**{f'amp{self.main_ch}':75e-3})
        seg_id_main = None
        loop_number = 0
        total_loops = len(dc_current)*len(pump_freqs)*self.params['n_main_amp']
        bar = tqdm()
        bar.reset(total=total_loops)
        for id_current, current in enumerate(dc_current):
            self.yoko.current = current
            for id_freq, freq in enumerate(pump_freqs):
                if seg_id_main:
                    self.pump.delete_segment(seg_id_main, ch=self.main_ch)
                seg_id_main = self.pump.send_sine(freq, 0, self.main_ch, 75e-3) #reset amp
                save_file_end_name = f"_current{id_current}_freq{id_freq}.h5"
                data_save_file_path = self.save_file_path[:-3]+save_file_end_name
                with h5.File(data_save_file_path, 'w') as file:
                    dset = file.create_dataset(
                        "data",
                        (self.params['n_main_amp'], len(freq_data), 2, 2),
                        dtype='complex128')
                    dset.dims[0].label = "Main pump amps"
                    dset.dims[1].label = "Vna frequencies (Hz)"
                    dset.dims[2].label = "X in SigXSigY"
                    dset.dims[3].label = "Y in SigXSigY"
                for id_main_amp, main_amp in enumerate(main_pump_amps[id_freq]):
                    if main_amp<75e-3:
                        raw_amp = 75e-3
                        fir_scale = main_amp/raw_amp
                    else:
                        raw_amp = main_amp
                        fir_scale = 1.0
                    self.pump.configure(
                        **{f'amp{self.main_ch}':raw_amp,
                            f'fir_scale{self.main_ch}':fir_scale}
                        )
                    self.pump.play(seg_id_main, self.main_ch)
                    self.pump.ask_if_done()
                    loop_number += 1
                    now = datetime.now()
                    time_str = now.strftime("%m/%d/%Y, %H:%M:%S")
                    self.vna.channel.run_averaging()
                    if loop_number==1:
                        self.vna.channel.autoscale_all(1)
                    data = np.array([
                        [self.vna.channel.read_raw_data('Sig1Sig1'),
                        self.vna.channel.read_raw_data('Sig1Sig2')],
                        [self.vna.channel.read_raw_data('Sig2Sig1'),
                        self.vna.channel.read_raw_data('Sig2Sig2')]
                        ]).transpose((2, 0, 1))
                    #saving
                    with h5.File(data_save_file_path, "r+") as file:
                        dset = file['data']
                        dset.attrs['time'] = time_str
                        dset[id_main_amp] = data
                    bar.update()
        bar.refresh()
        self.pump.clear_all_wfm()
        self.yoko.reset_config
            