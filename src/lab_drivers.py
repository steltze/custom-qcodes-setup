"""
QCoDeS drivers for instruments not covered by qcodes / qcodes_contrib_drivers.

  * KeysightP5024A     - Streamline VNA (subclass of the built-in PNA base)
  * AnaPicoAPUASYN20   - ultra-agile signal source (SCPI-1999)
  * KeysightM8195A     - 65 GSa/s AWG (AXIe, SCPI via the Soft Front Panel)

VERIFY-BEFORE-TRUST
-------------------
Every line marked `# VERIFY` encodes a spec or SCPI mnemonic taken from
datasheets / programming guides rather than from your actual hardware.
Check each against your instrument's manual (and `:SYST:ERR?`) before
relying on it.  Wrong *limits* are the dangerous ones - a too-generous
validator will happily let you program a value the hardware can't reach.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qcodes.instrument import VisaInstrument
from qcodes.instrument_drivers.Keysight.N52xx import KeysightPNAxBase
from qcodes.parameters import Parameter, create_on_off_val_mapping
from qcodes.validators import Enum, Numbers


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

    Address is usually::

        TCPIP0::localhost::hislip0::INSTR
        TCPIP0::localhost::inst0::INSTR

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


# ---------------------------------------------------------------------------
# 2. AnaPico APUASYN20
# ---------------------------------------------------------------------------
class AnaPicoAPUASYN20(VisaInstrument):
    """
    AnaPico APUASYN20 ultra-agile signal source (now sold by Keysight).

    Single-channel, 8 kHz - 20 GHz.  The multi-channel sibling is the
    APUASYN20-X; see ``channel`` below if you have one of those.

    SCPI-1999 over USB / Gb Ethernet / (optional) GPIB.
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


# ---------------------------------------------------------------------------
# 3. Keysight M8195A
# ---------------------------------------------------------------------------
class KeysightM8195A(VisaInstrument):
    """
    Keysight M8195A 65 GSa/s arbitrary waveform generator.

    The AXIe module itself appears as a *register-based* PXI resource and
    cannot be reached with SCPI.  Start the **M8195 Soft Front Panel**, which
    publishes a SCPI server, and connect to the address it reports - typically::

        TCPIP0::localhost::hislip0::INSTR

    Use Keysight IO Libraries VISA (``ResourceManager('@ivi')``) rather than
    pyvisa-py for this instrument.

    Scalar settings are plain SCPI; waveform download is binary and is the one
    part worth cross-checking against pyarbtools' implementation.
    """

    default_terminator = "\n"

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, **kwargs)

        self.dac_mode: Parameter = self.add_parameter(
            "dac_mode",
            label="DAC mode",
            get_cmd=":INST:DACM?",
            set_cmd=":INST:DACM {}",
            vals=Enum("SING", "DUAL", "FOUR", "MARK", "DCD", "DCM"),  # VERIFY
            docstring="SING/DUAL/FOUR = 1/2/4 channels; MARK/DCD/DCM = "
                      "marker and duplicate modes.",
        )

        self.sample_rate: Parameter = self.add_parameter(
            "sample_rate",
            label="Sample rate",
            unit="Sa/s",
            get_cmd=":FREQ:RAST?",
            set_cmd=":FREQ:RAST {:.6f}",
            get_parser=float,
            vals=Numbers(53.76e9, 65e9),   # VERIFY - depends on DAC mode
        )

        self.reference_source: Parameter = self.add_parameter(
            "reference_source",
            label="Reference clock source",
            get_cmd=":ROSC:SOUR?",
            set_cmd=":ROSC:SOUR {}",
            vals=Enum("EXT", "AXI", "INT"),   # VERIFY
        )

        self.trigger_source: Parameter = self.add_parameter(
            "trigger_source",
            label="Trigger source",
            get_cmd=":ARM:TRIG:SOUR?",
            set_cmd=":ARM:TRIG:SOUR {}",
            vals=Enum("TRIG", "EVEN", "INT"),  # VERIFY
        )

        # Per-channel parameters.  Kept flat (amplitude_1 ... amplitude_4)
        # rather than as submodules for simplicity; convert to
        # InstrumentChannel + ChannelList if you prefer.
        for ch in range(1, 5):
            self.add_parameter(
                f"amplitude_{ch}",
                label=f"Ch{ch} amplitude",
                unit="V",
                get_cmd=f":VOLT{ch}?",
                set_cmd=f":VOLT{ch} {{:.4f}}",
                get_parser=float,
                vals=Numbers(0, 1.0),        # VERIFY
            )
            self.add_parameter(
                f"offset_{ch}",
                label=f"Ch{ch} offset",
                unit="V",
                get_cmd=f":VOLT{ch}:OFFS?",
                set_cmd=f":VOLT{ch}:OFFS {{:.4f}}",
                get_parser=float,
            )
            self.add_parameter(
                f"output_{ch}",
                label=f"Ch{ch} output enabled",
                get_cmd=f":OUTP{ch}?",
                set_cmd=f":OUTP{ch} {{}}",
                val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
            )

        self.connect_message()

    # -- run control --------------------------------------------------------
    def run(self) -> None:
        """Start signal generation."""
        self.write(":INIT:IMM")

    def stop(self) -> None:
        """Stop signal generation."""
        self.write(":ABOR")

    # -- waveform handling --------------------------------------------------
    def upload_waveform(
        self,
        waveform: np.ndarray,
        channel: int = 1,
        segment: int = 1,
    ) -> None:
        """
        Download a waveform to a segment.

        `waveform` should be float in [-1, 1]; it is scaled to signed 8-bit,
        which is the M8195A's native DAC resolution.

        The granularity / minimum-length rules depend on DAC mode and are the
        most common source of "Data out of range" errors.  If this misbehaves,
        compare against ``pyarbtools.instruments.M8195A.download_wfm``, which
        already handles the padding rules (note pyarbtools is GPL-3).
        """
        wfm = np.asarray(waveform, dtype=float)
        if np.max(np.abs(wfm)) > 1.0:
            raise ValueError("waveform must be normalised to [-1, 1]")

        data = np.int8(np.round(wfm * 127))

        gran = 256          # VERIFY - depends on DAC mode
        min_len = 1280      # VERIFY
        if len(data) < min_len:
            data = np.concatenate([data, np.zeros(min_len - len(data), dtype=np.int8)])
        if len(data) % gran:
            pad = gran - (len(data) % gran)
            data = np.concatenate([data, np.zeros(pad, dtype=np.int8)])

        self.write(f":TRAC{channel}:DEF {segment},{len(data)},0")
        self.visa_handle.write_binary_values(
            f":TRAC{channel}:DATA {segment},0,",
            data,
            datatype="b",
        )
        self.check_error()

    def delete_all_waveforms(self, channel: int = 1) -> None:
        self.write(f":TRAC{channel}:DEL:ALL")

    # -- diagnostics --------------------------------------------------------
    def check_error(self) -> None:
        """Raise if the instrument's error queue is non-empty."""
        err = self.ask(":SYST:ERR?")
        if not err.startswith(("0,", "+0,")):
            raise RuntimeError(f"{self.name} reported: {err}")