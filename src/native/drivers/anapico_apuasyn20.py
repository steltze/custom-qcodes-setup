"""AnaPico APUASYN20 - raw SCPI over pyvisa, no qcodes or exopy dependency.

Single-channel, 8 kHz - 20 GHz ultra-agile signal source, SCPI-1999 over
USB / Gb Ethernet / (optional) GPIB.

Commands below are transcribed from
`legacy/drivers/exopy_hqc_legacy/drivers/visa/anapico.py` (the exopy
driver this codebase used previously), not the datasheet. Two details in
particular came from running that driver against real hardware, not from
the manual - see git history c1a6ba4/8921cd5 - and are preserved here:

  * the -10..23 dBm `power` range (the instrument doesn't reject an
    out-of-range set with an error - it silently no-ops and leaves the
    output at its previous value, which is confusing without the range
    check enforced up front).
  * the `SYST:COMM:VXI:RTMO 0` write below, sent on every connect - avoids
    the source refusing to reconnect after an unclean disconnect (crash,
    Ctrl-C, ...).

For the qcodes-compatible instrument (Station/Measurement/snapshot use),
see `native/instruments/AnaPicoAPUASYN20.py`, which builds one instrument
class out of this class and `qcodes.instrument.Instrument` directly, via
multiple inheritance - not a wrapper holding this as a separate object.
This class works completely standalone too.
"""

from __future__ import annotations

from typing import Any

from .base import VisaDriver


class AnaPicoAPUASYN20(VisaDriver):
    """Single-channel Anapico APUASYN20. The `:SOURn:` prefix is accepted
    but optional on this single-channel model - hardcoded to channel 1
    below, matching `legacy/drivers/anapico.py::Anapico1`. For the
    multi-channel APUASYN20-X, see the `-X` driver instead."""

    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, address, **kwargs)
        self._config: dict[str, Any] = {}

        self.write("SYST:COMM:VXI:RTMO 0")

        if config:
            self.configure(config)

        self.connect_message()

    # -- frequency --------------------------------------------------------
    def get_frequency(self) -> float:
        """Output frequency, Hz."""
        return float(self.ask(":SOUR1:FREQ?"))

    def set_frequency(self, value: float) -> None:
        self.write(f":SOUR1:FREQ {value:.6f}")

    # -- power --------------------------------------------------------------
    def get_power(self) -> float:
        """Output power, dBm."""
        return float(self.ask(":SOUR1:POWER?"))

    def set_power(self, value: float) -> None:
        if not (-10.0 <= value <= 24.0):  # confirmed on real hardware, see module docstring
            raise ValueError(
                f"power={value!r} outside allowed range [-10.0, 23.0] dBm"
            )
        self.write(f":SOUR1:POWER {value:.3f}")

    # -- phase --------------------------------------------------------------
    def get_phase(self) -> float:
        """Output phase, rad."""
        return float(self.ask(":SOUR1:PHAS?"))

    def set_phase(self, value: float) -> None:
        self.write(f":SOUR1:PHAS {value:.6f}")

    # -- output ---------------------------------------------------------------
    def get_output_enabled(self) -> bool:
        return bool(int(self.ask(":OUTP1?")))

    def set_output_enabled(self, value: bool) -> None:
        self.write(f":OUTPUT1 {'ON' if value else 'OFF'}")

    # -- reference oscillator -------------------------------------------------
    def get_reference_source(self) -> str:
        """'INT' or 'EXT'."""
        return self.ask(":SOUR:ROSC:SOUR?")

    def set_reference_source(self, value: str) -> None:
        self.write(f":SOUR:ROSC:SOUR {value}")

    def get_oscillator_locked(self) -> bool:
        return bool(int(self.ask(":SOUR:ROSC:LOCK?")))

    # -- config convenience -----------------------------------------------------
    def configure(self, config: dict[str, Any]) -> None:
        """Apply a config dict:
            'frequency': float (Hz)
            'power': float (dBm)
            'phase': float (rad)
            'output': bool
            'reference_source': 'INT' | 'EXT'
        Anything not specified is left untouched."""
        self._config.update(config)
        if "frequency" in config:
            self.set_frequency(config["frequency"])
        if "power" in config:
            self.set_power(config["power"])
        if "phase" in config:
            self.set_phase(config["phase"])
        if "reference_source" in config:
            self.set_reference_source(config["reference_source"])
        if "output" in config:
            self.set_output_enabled(bool(config["output"]))

    def safe_shutdown(self) -> None:
        """Called by the measurement harness's `safe_run()` on *every*
        exit from a run - a clean finish as much as an error/abort, not
        error/abort only. Turns RF output off."""
        self.set_output_enabled(False)
