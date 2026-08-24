"""AnaPico APUASYN20-X (4 channels) - raw SCPI over pyvisa, no qcodes or
exopy dependency.

Sibling of `native.drivers.anapico_apuasyn20.AnaPicoAPUASYN20` (single-channel) -
see that module for the channel-less equivalent of every method below and
for where the SCPI dialect and the real-hardware notes (power range,
`SYST:COMM:VXI:RTMO 0`) came from; both are transcribed from the same
`legacy/drivers/exopy_hqc_legacy/drivers/visa/anapico.py::
AnapicoNChannels`/`AnapicoChannel`, which is channel-count-generic.

Channel numbers are 1-indexed throughout (`channel=1..4`), matching every
other multi-channel instrument in this repo (e.g. `KeysightM8195A`'s
`amplitude_1`.. `amplitude_4`) and the SCPI itself (`:SOUR1:FREQ?` ..
`:SOUR4:FREQ?`).

For the qcodes-compatible instrument, see
`native/instruments/AnaPicoAPUASYN20X.py`.
"""

from __future__ import annotations

from typing import Any

from .base import VisaDriver

CHANNELS = (1, 2, 3, 4)


def _parse_on_off(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() in ("ON", "1")
    return bool(value)


class AnaPicoAPUASYN20X(VisaDriver):
    """4-channel Anapico APUASYN20-X, 8 kHz - 20 GHz per channel."""

    default_terminator = "\n"
    channels = CHANNELS

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

    # -- per-channel parameters -------------------------------------------------
    def get_frequency(self, channel: int) -> float:
        """Output frequency, Hz."""
        return float(self.ask(f":SOUR{channel}:FREQ?"))

    def set_frequency(self, channel: int, value: float) -> None:
        self.write(f":SOUR{channel}:FREQ {value:.6f}")

    def get_power(self, channel: int) -> float:
        """Output power, dBm."""
        return float(self.ask(f":SOUR{channel}:POWER?"))

    def set_power(self, channel: int, value: float) -> None:
        # -10..23 dBm confirmed on real hardware for the single-channel
        # unit (see native/drivers/anapico_apuasyn20.py) - VERIFY this holds on
        # all 4 channels of your -X unit, not just channel 1.
        if not (-10.0 <= value <= 23.0):
            raise ValueError(
                f"power={value!r} outside allowed range [-10.0, 23.0] dBm"
            )
        self.write(f":SOUR{channel}:POWER {value:.3f}")

    def get_phase(self, channel: int) -> float:
        """Output phase, rad."""
        return float(self.ask(f":SOUR{channel}:PHAS?"))

    def set_phase(self, channel: int, value: float) -> None:
        self.write(f":SOUR{channel}:PHAS {value:.6f}")

    def get_output_enabled(self, channel: int) -> bool:
        return bool(int(self.ask(f":OUTP{channel}?")))

    def set_output_enabled(self, channel: int, value: bool) -> None:
        self.write(f":OUTPUT{channel} {'ON' if value else 'OFF'}")

    def which_outputs_enabled(self) -> dict[int, bool]:
        """`{channel: is_output_enabled}` for all 4 channels in one call."""
        return {ch: self.get_output_enabled(ch) for ch in self.channels}

    # -- shared across all channels ---------------------------------------------
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
            'frequencies': (float, float, float, float), Hz
            'powers': (float, float, float, float), dBm
            'phases': (float, float, float, float), rad
            'outputs': (bool | 'ON' | 'OFF', ...) x4
            'reference_source': 'INT' | 'EXT'
        Each 4-tuple maps positionally onto channels 1..4. Anything not
        specified is left untouched."""
        self._config.update(config)
        if "frequencies" in config:
            for ch, value in zip(self.channels, config["frequencies"]):
                self.set_frequency(ch, value)
        if "powers" in config:
            for ch, value in zip(self.channels, config["powers"]):
                self.set_power(ch, value)
        if "phases" in config:
            for ch, value in zip(self.channels, config["phases"]):
                self.set_phase(ch, value)
        if "reference_source" in config:
            self.set_reference_source(config["reference_source"])
        if "outputs" in config:
            for ch, value in zip(self.channels, config["outputs"]):
                self.set_output_enabled(ch, _parse_on_off(value))

    def safe_shutdown(self) -> None:
        """Called by the measurement harness's `safe_run()` on *every*
        exit from a run - a clean finish as much as an error/abort, not
        error/abort only. Turns RF output off on all 4 channels."""
        for ch in self.channels:
            self.set_output_enabled(ch, False)
