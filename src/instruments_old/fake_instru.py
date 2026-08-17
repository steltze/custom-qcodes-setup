from copy import deepcopy

from .basic_instrument import BasicInstrument

class FakeInstru(BasicInstrument):
    """
    Fake instrument for testing code.
    """
    
    #default config
    default_config = {
        'powers':(-10.0, -10.0, -10.0, -10.0),
        'frequencies':(1e9, 1e9, 1e9, 1e9),
        'outputs':("OFF", "OFF", "OFF", "OFF"),
        'low_noise':("ON", "ON", "ON", "ON"),
        'reference_osc':"EXT"
    }
    
    #no overwrite