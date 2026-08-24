from .basic_instrument import BasicInstrument
from .exopy_hqc_legacy.drivers.visa.signal_hound_sa124b import SA124B

class SignalHoundSA(BasicInstrument, SA124B):
    """Customized object interfacing to custom visa driver.
    Main goal is to avoid misconfiguration. Also standardizes
    configuration saving.
    """
    
    #default config
    default_config = {
        'mode':"SA",
        'initiate':0,
        'bandwidth':250e3,
        'frequency':4e9,
        'span':200e3,
        'reference_level':0,
        'reference_osc':"EXT",
        'ZS_time':0.01
    }

    def __init__(self, visa_address, nickname, config=None):
        BasicInstrument.__init__(self, visa_address, nickname, config)
        SA124B.__init__(self, self.connection_info)
        self.preset()
        self.set_config(self.config)
    
    def set_config(self, config):
        """config should be a dictionnary with
        'mode':"SA" or "ZS",
        'initiate':0 or 1,
        'bandwidth':float (Hz),
        'frequency':float (Hz),
        'span':float (Hz),
        'reference_level':float (dBm),
        'reference_osc':"EXT" or "INT"
        """
        self.config = config
        if 'mode' in config.keys():
            self.mode = config['mode']
        if 'initiate' in config.keys():
            self.initiate = config['initiate']
        if 'frequency' in config.keys():
            self.frequency = config['frequency']
        if 'span' in config.keys():
            self.span = config['span']
        if 'bandwidth' in config.keys():
            self.bandwidth = config['bandwidth']
        if 'reference_level' in config.keys():
            self.ref_level = config['reference_level']
        if 'reference_osc' in config.keys():
            self.ref_oscillator = config['reference_osc']
        if 'ZS_time' in config.keys():
            self.zs_time = config['ZS_time']
            
    def get_sa_trace(self):
        self.fire_trigger()
        self.ask_if_done()
        xstart = self.get_sa_xstart()
        xinc = self.get_sa_xinc()
        data = self.get_sa_data()
        return ((xstart, xinc), data)