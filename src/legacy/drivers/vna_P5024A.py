from .basic_instrument import BasicInstrument
from .exopy_hqc_legacy.drivers.visa.agilent_pna import AgilentPNA

class VNA_P5024A(BasicInstrument, AgilentPNA):
    """Customized object interfacing to exopy legacy driver.
    Main goal is to avoid misconfiguration. Also standardizes
    configuration saving.
    """
    
    #default config
    default_config = {
        'sweep_type':"LIN",
        'measurements':(("default trace", "S11"),),
        'power':-40.0,
        'if_bandwidth':5000,
        'freq_spec':(10e8, 20e9, 2001),
        'average':1
    }
    
    def __init__(self, visa_address, nickname, config=None):
        BasicInstrument.__init__(self, visa_address, nickname, config)
        AgilentPNA.__init__(self, self.connection_info)
        self.reset_config()
        self.set_config(self.config)
            
    def reset_config(self):
        """Presets the VNA, then sets to hold,
        on and no traces"""
        self.preset()
        self.set_all_chanel_to_hold()
        self.output = 1
        self.data_format = 'REAL,64'
        self.channel = self.get_channel(1)
        self.channel.delete_meas("CH1_S11_1")
        
    def set_config(self, config):
        """config should be a dictionnary with
        'sweep_type': type of sweep ('LIN' or 'CW')
        'measurements' : list of measurements to add (name:str, Sparam:str)
        'power' : channel power:float
        'if_bandwidth' : channel bandwidth:int
        'freq_spec' : (freq_start:float, freq_stop:float, n_freqs:int)
        'average' : averaging number:int
        Anything not specified is not changed.
        """
        self.config.update(config)
        if 'sweep_type' in config.keys():
            self.channel.sweep_type = config['sweep_type']
        if 'freq_spec' in config.keys():
            if self.config["sweep_type"]=="LIN":
                (freq_start, freq_stop, n_freqs) = config['freq_spec']
                self.channel.frequency_start = freq_start
                self.channel.frequency_stop = freq_stop
                self.channel.sweep_points = n_freqs
            elif self.config["sweep_type"]=="CW":
                (freq_cw, n_pts) = config['freq_spec']
                self.channel.frequency_cw = freq_cw
                self.channel.sweep_points = n_pts
            else:
                raise ValueError(f'Unknown sweep_type : {self.config["sweep_type"]}')
        if 'measurements' in config.keys():
            for (i_meas, (name, S_param)) in enumerate(config['measurements']):
                self.channel.create_meas(name, S_param)
                self.channel.bind_meas_to_window(name, 1, i_meas+1)
        self.channel.couple_scale(1) #force use first window
        if 'power' in config.keys():
            self.channel.power = config['power']
        if 'if_bandwidth' in config.keys():
            self.channel.if_bandwidth = config['if_bandwidth']
        if 'average' in config.keys():
            n_avg = config['average']
            if n_avg==1:
                self.channel.average_state = 0
            else:
                self.channel.average_state = 1
                self.channel.average_count = n_avg