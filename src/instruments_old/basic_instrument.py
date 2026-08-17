from copy import deepcopy

class BasicInstrument():
    """Common base for interfacing to exopy_legacy_drivers.
    Contains the code to standardize config, and for writing 
    to save file.
    """
    
    #default config, should be overwritten
    default_config = {
    }
    
    def __init__(self, visa_address, nickname, config=None):
        self.visa_address = visa_address
        self.nickname = nickname
        self.config = deepcopy(self.default_config)
        #overwrite default_config with specified one
        if config is not None:
            for key in self.default_config.keys():
                if key in config.keys():
                    self.config[key] = config[key]
        self.connection_info = {'resource_name':self.visa_address}
            
    def get_config_info(self):
        '''Get config and informations for saving purposes.
        Returns as a dictionnary.'''
        info = {'visa_address':self.visa_address,
                'nickname':self.nickname,
                'config':self.config}
        return info