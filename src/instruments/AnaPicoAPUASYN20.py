# ---------------------------------------------------------------------------
# AnaPico APUASYN20
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np

from qcodes.instrument import VisaInstrument
from qcodes.instrument_drivers.Keysight.N52xx import KeysightPNAxBase
from qcodes.parameters import Parameter, create_on_off_val_mapping
from qcodes.validators import Enum, Numbers


class AnaPicoAPUASYN20(VisaInstrument):
    """
    AnaPico APUASYN20 ultra-agile signal source (now sold by Keysight).

    Single-channel, 8 kHz - 20 GHz.  The multi-channel sibling is the
    APUASYN20-X; see ``channel`` below if you have one of those.

    """

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        channel: int = 1,
        min_freq: float = 8e3,      # VERIFY
        max_freq: float = 20e9,     # VERIFY
        min_power: float = -20,     # VERIFY - depends on options
        max_power: float = 20,      # VERIFY - depends on options
        **kwargs: Any,
    ) -> None:
        super().__init__(name, address, **kwargs)

        self._ch = channel
        # On the single-channel APUASYN20 the ':SOURn:' prefix is accepted but
        # optional.  On the -X you need it.  VERIFY against your manual.
        p = f":SOUR{channel}"

        self.frequency: Parameter = self.add_parameter(
            "frequency",
            label="Frequency",
            unit="Hz",
            get_cmd=f"{p}:FREQ?",
            set_cmd=f"{p}:FREQ {{:.6f}}",
            get_parser=float,
            vals=Numbers(min_freq, max_freq),
        )

        self.power: Parameter = self.add_parameter(
            "power",
            label="Output power",
            unit="dBm",
            get_cmd=f"{p}:POW?",
            set_cmd=f"{p}:POW {{:.3f}}",
            get_parser=float,
            vals=Numbers(min_power, max_power),
        )

        self.phase: Parameter = self.add_parameter(
            "phase",
            label="Phase",
            unit="rad",
            get_cmd=f"{p}:PHAS?",
            set_cmd=f"{p}:PHAS {{:.6f}}",
            get_parser=float,
            vals=Numbers(-np.pi, np.pi),   # VERIFY - some firmware uses degrees
        )

        self.output_enabled: Parameter = self.add_parameter(
            "output_enabled",
            label="Output enabled",
            get_cmd=f":OUTP{channel}:STAT?",
            set_cmd=f":OUTP{channel}:STAT {{}}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )

        self.reference_source: Parameter = self.add_parameter(
            "reference_source",
            label="Reference oscillator source",
            get_cmd=":ROSC:SOUR?",
            set_cmd=":ROSC:SOUR {}",
            vals=Enum("INT", "EXT"),       # VERIFY
        )

        self.reference_frequency: Parameter = self.add_parameter(
            "reference_frequency",
            label="External reference frequency",
            unit="Hz",
            get_cmd=":ROSC:EXT:FREQ?",
            set_cmd=":ROSC:EXT:FREQ {:.0f}",
            get_parser=float,
        )

        self.connect_message()

    def get_error(self) -> str:
        """Pop one entry off the instrument's error queue."""
        return self.ask(":SYST:ERR?")

    def flush_errors(self) -> list[str]:
        """Drain the error queue; returns everything that was queued."""
        errors = []
        for _ in range(50):
            err = self.get_error()
            if err.startswith(("0,", "+0,")):
                break
            errors.append(err)
        return errors


