from .basic_instrument import BasicInstrument
from .exopy_hqc_legacy.drivers.visa.yokogawa import YokogawaGS200

class Yokogs200(BasicInstrument, YokogawaGS200):
    """Customized object interfacing to (highly modified) 
    exopy legacy driver. Main goal is to avoid misconfiguration. Also standardizes
    configuration saving.
    """
    #default config
    default_config = {
            'mode': 'CURR',
            'current_value': 0,
            'voltage_value': 0,
            'voltage_range': '1 V',
            'current_range': '1 mA',
            'ramp_step': 1e-6,
            'delay': 500e-6,
            'output': 'on'
        }
    
    def __init__(self, visa_address, nickname, config=None):
        BasicInstrument.__init__(self, visa_address, nickname, config)
        YokogawaGS200.__init__(self, self.connection_info)
        # self.open_connection()
        self.reset_config() 
        self.set_config(self.config)

    def reset_config(self):
        self.output = 'off'
        # self.voltage = 0.0
        # self.function = 'CURR'
        # self.current = 0.0
        
    def set_config(self, config):
        """config should be a dictionnary with
        'current_value' : [A] float
        'voltage_limit' : str in ['10 mV', '100 mV', '1 V', '10 V', '30 V']
        'current_range' : str in ['1 mA', '10 mA', '100 mA', '200 mA']
        'ramp_step': [s] float,
        'delay': [s] float,
        Anything not specified is not changed.
        """

        if "mode" in config.keys() and config['mode'] == 'VOLT':
            self.function = 'VOLT'
            if 'voltage_range' in config.keys():
                self.voltage_range = config['voltage_range']
            if 'voltage_value' in config.keys():
                self.voltage = config['voltage_value']
        else:
            self.function = 'CURR'
            if 'current_range' in config.keys():
                self.current_range = config['current_range']
            if 'current_value' in config.keys():
                self.current = config['current_value']

        if 'output' in config.keys():
            self.output = config['output']


    def ramp_voltage(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the voltage from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Volt
            step: The ramp steps in Volt
            delay: The time between finishing one step and
                starting another in seconds.

        """
        self._assert_mode("VOLT")
        self._ramp_source(ramp_to, step, delay)

    def ramp_current(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the current from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Ampere
            step: The ramp steps in Ampere
            delay: The time between finishing one step and starting
                another in seconds.

        """
        self._assert_mode("CURR")
        self._ramp_source(ramp_to, step, delay)



    def _ramp_source(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the output from the current level to the specified output

        Args:
            ramp_to: The ramp target in volts/amps
            step: The ramp steps in volts/ampere
            delay: The time between finishing one step and
                starting another in seconds.

        """
        saved_step = self.output_level.step
        saved_inter_delay = self.output_level.inter_delay

        self.output_level.step = step
        self.output_level.inter_delay = delay
        self.output_level(ramp_to)

        self.output_level.step = saved_step
        self.output_level.inter_delay = saved_inter_delay

    def _assert_mode(self, mode) -> None:
        """
        Assert that we are in the correct mode to perform an operation.

        Args:
            mode: "CURR" or "VOLT"

        """
        if self.function != mode:
            raise ValueError(
                f"Cannot get/set {mode} settings while in {self.function} mode"
            )