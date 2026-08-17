# ---------------------------------------------------------------------------
# Keysight M8195A
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np

from qcodes.instrument import VisaInstrument
from qcodes.parameters import Parameter, create_on_off_val_mapping
from qcodes.validators import Enum, Numbers


class KeysightM8195A(VisaInstrument):
    """
    Keysight M8195A 65 GSa/s arbitrary waveform generator.

    The AXIe module itself appears as a *register-based* PXI resource and
    cannot be reached with SCPI.  Start the **M8195 Soft Front Panel**, which
    publishes a SCPI server, and connect to the address it reports - typically::

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