import os
import numpy as np
import h5py as h5
from datetime import datetime
from copy import deepcopy
from tqdm.notebook import tqdm

from time import sleep

from .base_measurement import BaseMeasurement
from ..instruments.vna_P5024A import VNA_P5024A

class VNACalibSlopeCustomMeas(BaseMeasurement):
    def __init__(self, save_path, circuit_path,
                 vna_params, sweep_params):
        '''
        vna_params:dict('visa_address':str, 'nickname':str, config:dict)
        'sweep_params':dict(
                'pts_list': list(int),
                'bw_list': list(int),
                'power_slope':float,
                'measurements': for example (
            ('Sig1Sig1', 'S31'),
            ('Sig2Sig1', 'S41'),
            ('Sig1Sig2', 'S32'),
            ('Sig2Sig2', 'S42')       
        )
                )
            )
        '''
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        self.vna = VNA_P5024A(**vna_params_mod)
        self.instruments = (self.vna,)
        #call the super at the end!
        super().__init__(save_path, circuit_path)
        
    def execute(self):
        n_pts = self.params['pts_list']
        bw_list = self.params['bw_list']
        power_slope = self.params['power_slope']
        self.vna.channel.power_slope_state = 1
        self.vna.channel.power_slope = power_slope
        
        bar = tqdm()
        bar.reset(total=len(n_pts))
        for i in range(len(n_pts)):
            n_pt = n_pts[i]
            bw = bw_list[i]
            self.vna.channel.if_bandwidth = bw
            self.vna.channel.sweep_points = n_pt
            self.vna.channel.run_averaging()
            freq_data = self.vna.channel.read_freq_data()
            data = np.array([
                [self.vna.channel.read_raw_data('Sig1Sig1'),
                self.vna.channel.read_raw_data('Sig1Sig2')],
                [self.vna.channel.read_raw_data('Sig2Sig1'),
                self.vna.channel.read_raw_data('Sig2Sig2')]
                ]).transpose((2, 0, 1))
            file_path = self.save_file_path[:-3] + f"_{n_pt}pts.h5"
            with h5.File(file_path, "w") as file:
                data_group = file.create_group('data')
                data_group.create_dataset("Vna frequencies (Hz)", data=freq_data)
                dset = data_group.create_dataset(
                    "data",
                    data=data,
                    dtype='complex128')
                dset.dims[0].label = "Vna frequencies (Hz)"
                dset.dims[1].label = "X in SigXSigY"
                dset.dims[2].label = "Y in SigXSigY"
            bar.update()
        bar.refresh()
            