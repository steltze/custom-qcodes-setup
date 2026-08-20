# ---------------------------------------------------------------------------
# Yokogawa GS200
# ---------------------------------------------------------------------------
#
# Unlike the AWG/Anapico, no legacy wrapper is needed here: QCoDeS ships a
# fully-featured native `YokogawaGS200` driver that already covers what
# `instruments_old/Yokogs200.py` hand-rolled (mode switching, ranges,
# output, and ramping via `output_level.step`/`.inter_delay` - the exact
# mechanism `instruments_old/Yokogs200.py::_ramp_source` reimplemented by
# hand). This subclass only adds the same `config`-dict vocabulary the old
# driver used, so call sites barely change.

from __future__ import annotations

from typing import Any

from qcodes.instrument_drivers.yokogawa.Yokogawa_GS200 import (
    YokogawaGS200 as _NativeYokogawaGS200,
)

# The old config dicts specify ranges as human strings ('1 mA', '10 V', ...);
# the native driver's current_range/voltage_range parameters want plain
# floats in SI units (checked against the installed qcodes driver: Enum(1e-3,
# 10e-3, 100e-3, 200e-3) for current, Enum(10e-3, 100e-3, 1, 10, 30) for
# voltage). Order matters: check "mV"/"mA" before "V"/"A".
_RANGE_UNIT_SCALE = {"mV": 1e-3, "mA": 1e-3, "V": 1.0, "A": 1.0}


def _parse_range(value: float | str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    for suffix in ("mV", "mA", "V", "A"):
        if text.endswith(suffix):
            return float(text[: -len(suffix)].strip()) * _RANGE_UNIT_SCALE[suffix]
    raise ValueError(f"Unrecognized range string: {value!r}")


class YokogawaGS200(_NativeYokogawaGS200):
    """QCoDeS-native Yokogawa GS200, with the same `config` dict shape
    `instruments_old/Yokogs200.py` used:
        'mode': 'CURR' | 'VOLT'
        'current_value' / 'voltage_value': float
        'current_range' / 'voltage_range': str (e.g. '1 mA', '10 V') or float
        'output': 'on' | 'off'
    Anything not specified is left untouched. Note: like the old driver,
    `ramp_step`/`delay` are accepted in the config dict but not applied here
    - none of the measurement scripts ramp through `configure()`, they set
    `current`/`voltage` directly at each sweep point, matching the old
    driver's actual (not just documented) behaviour.
    """

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # The native driver hardcodes `default_terminator = "\n"` (LF
        # only). Confirmed on real hardware: this GS200 terminates its
        # replies with CR+LF, so an LF-only read strips just the "\n" and
        # leaves a trailing "\r" stuck on every parsed value (e.g.
        # `source_mode()` returning "CURR\r") - invisible for plain
        # writes/one-off reads, but breaks the native driver's own
        # internal `_assert_mode()` strict-equality check the moment
        # anything (e.g. `current_range()`) is read back as a getter,
        # which is exactly the code path the old scripts never exercised
        # (they only ever *set* this instrument's properties). Overridable
        # via `terminator=...` if a different setup needs something else.
        kwargs.setdefault("terminator", "\r\n")
        super().__init__(name, address, **kwargs)
        self._visa_address = address
        self._config = dict(config) if config else {}

        self.reset_config()
        if self._config:
            self.configure(self._config)

    def get_config_info(self) -> dict[str, Any]:
        """Same shape as `instruments_old.basic_instrument.BasicInstrument
        .get_config_info`, so this instrument can be dropped straight into
        `BaseMeasurement.instruments`."""
        return {
            "visa_address": self._visa_address,
            "nickname": self.name,
            "config": self._config,
        }

    def reset_config(self) -> None:
        """Safe state: output off. Matches
        `instruments_old/Yokogs200.py::reset_config`."""
        self.output("off")

    def safe_shutdown(self) -> None:
        """Called by the measurement harness on any error/abort - same
        safe state as `reset_config`."""
        self.reset_config()

    def configure(self, config: dict[str, Any]) -> None:
        """Apply a config dict. See class docstring for keys."""
        if config.get("mode") == "VOLT":
            self.source_mode("VOLT")
            if "voltage_range" in config:
                self.voltage_range(_parse_range(config["voltage_range"]))
            if "voltage_value" in config:
                self.voltage(config["voltage_value"])
        else:
            self.source_mode("CURR")
            if "current_range" in config:
                self.current_range(_parse_range(config["current_range"]))
            if "current_value" in config:
                self.current(config["current_value"])

        if "output" in config:
            self.output(config["output"])
