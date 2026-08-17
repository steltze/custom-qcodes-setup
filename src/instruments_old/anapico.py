from .basic_instrument import BasicInstrument
from .exopy_hqc_legacy.drivers.visa.anapico import AnapicoNChannels

from time import sleep

class Anapico4(BasicInstrument, AnapicoNChannels):
    """Customized object interfacing to (highly modified) 
    exopy legacy driver. Main goal is to avoid misconfiguration. Also standardizes
    configuration saving.
    """
    
    #default config
    default_config = {
        'powers':(-10.0, -10.0, -10.0, -10.0),
        'frequencies':(1e9, 1e9, 1e9, 1e9),
        'phases':(0.0, 0.0, 0.0, 0.0),
        'outputs':("OFF", "OFF", "OFF", "OFF"),
        'reference_osc':"EXT"
    }
    
    def __init__(self, visa_address, nickname, config=None):
        BasicInstrument.__init__(self, visa_address, nickname, config)
        AnapicoNChannels.__init__(self, self.connection_info)
        self.set_config(self.config)
    
    def set_config(self, config):
        """config should be a dictionnary with
        'powers' : (float, float, float, float)
            list of powers for the 4 channels
        'frequencies': (float, float, float, float)
            list of frequencies for the 4 channels,
        'phases': (float, float, float, float)
            list of phases for the 4 channels,
        'outputs': (str, str, str, str)
            list of output states for the 4 channels,
        'reference_osc': str
            reference oscillator source
        """
        self.config = config
        if 'outputs' in config.keys():
            for i_ch, output in enumerate(config['outputs']):
                self.channels[i_ch].output = output
        if 'powers' in config.keys():
            for i_ch, power in enumerate(config['powers']):
                self.channels[i_ch].power = power
        if 'frequencies' in config.keys():
            for i_ch, freq in enumerate(config['frequencies']):
                self.channels[i_ch].frequency = freq
        if 'phases' in config.keys():
            for i_ch, phase in enumerate(config['phases']):
                self.channels[i_ch].phase = phase
        if 'reference_osc' in config.keys():
            self.ref_oscillator = config['reference_osc']
            if self.ref_oscillator == "EXT":
                sleep(0.5)
                assert(self.oscillator_lock)
                
class Anapico1(BasicInstrument, AnapicoNChannels):
    """Customized object interfacing to (highly modified) 
    exopy legacy driver. Main goal is to avoid misconfiguration. Also standardizes
    configuration saving.
    """
    
    #default config
    default_config = {
        'powers':(-10.0,),
        'frequencies':(1e9,),
        'phases':(0.0,),
        'outputs':("OFF",),
        'reference_osc':"EXT"
    }
    
    def __init__(self, visa_address, nickname, config=None):
        BasicInstrument.__init__(self, visa_address, nickname, config)
        AnapicoNChannels.__init__(self, self.connection_info, n_channels=1)
        self.set_config(self.config)
    
    def set_config(self, config):
        """config should be a dictionnary with
        'powers' : (float, float, float, float)
            list of powers for the 4 channels
        'frequencies': (float, float, float, float)
            list of frequencies for the 4 channels,
        'phases': (float, float, float, float)
            list of phases for the 4 channels,
        'outputs': (str, str, str, str)
            list of output states for the 4 channels,
        'reference_osc': str
            reference oscillator source
        """
        self.config = config
        if 'outputs' in config.keys():
            for i_ch, output in enumerate(config['outputs']):
                self.channels[i_ch].output = output
        if 'powers' in config.keys():
            for i_ch, power in enumerate(config['powers']):
                self.channels[i_ch].power = power
        if 'frequencies' in config.keys():
            for i_ch, freq in enumerate(config['frequencies']):
                self.channels[i_ch].frequency = freq
        if 'phases' in config.keys():
            for i_ch, phase in enumerate(config['phases']):
                self.channels[i_ch].phase = phase
        if 'reference_osc' in config.keys():
            self.ref_oscillator = config['reference_osc']
            if self.ref_oscillator == "EXT":
                sleep(0.5)
                assert(self.oscillator_lock)

