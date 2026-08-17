from __future__ import annotations

from typing import Any

import numpy as np

from qcodes.instrument_drivers.Keysight.N52xx import KeysightPNAxBase


# ---------------------------------------------------------------------------
# 1. Keysight P5024A Streamline VNA
# ---------------------------------------------------------------------------
class KeysightP5024A(KeysightPNAxBase):
    """
    Keysight P5024A Streamline USB VNA.

    The Streamline series runs the same firmware as the PNA family, so the
    stock ``KeysightPNAxBase`` does all the work.  This subclass only supplies
    the hardware limits.

    NOTE: the *Network Analyzer application must be running on the host PC*
    before any SCPI interface exists - the instrument itself is faceless.

    To confirm the limits from the instrument itself::

        inst.ask(':SENS:FREQ:STAR? MIN')
        inst.ask(':SENS:FREQ:STOP? MAX')
        inst.ask(':SOUR:POW? MIN')
        inst.ask(':SOUR:POW? MAX')
        inst.ask(':SYST:CAP:HARD:PORT:COUN?')   # number of ports
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(
            name,
            address,
            min_freq=9e3,      # VERIFY
            max_freq=20e9,     # VERIFY
            min_power=-100,    # VERIFY
            max_power=13,      # VERIFY
            nports=4,          # VERIFY - P502xA is the 4-port line
            **kwargs,
        )