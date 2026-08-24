# ---------------------------------------------------------------------------
# Signal Hound USB SA124B
# ---------------------------------------------------------------------------
#
# Unlike every other instrument in this folder, QCoDeS' native driver for
# this one talks to the device through Signal Hound's own `sa_api` SDK via
# ctypes (`saOpenDevice`/`saConfigCenterSpan`/`saGetSweep_32f`/...), not
# VISA/SCPI - so there is no `visa_address`, and the old exopy driver
# (legacy/drivers/exopy_hqc_legacy/drivers/visa/signal_hound_sa124b.py),
# which talked SCPI to Signal Hound's "Spike" software over VISA, has no
# real overlap with it below the parameter surface. This subclass adds the
# same `config`-dict convenience the rest of this folder uses, mapped onto
# whatever the native driver actually supports - see the two NOTEs on the
# class docstring for what does not carry over.
#
# Requires Signal Hound's Spike software/SDK installed and `sa_api.dll` (or
# the platform equivalent) reachable on this machine - VERIFY `dll_path`
# for your setup; the qcodes default (`SignalHoundUSBSA124B.dll_path`) is a
# Windows path. Not constructible/testable on this repo's Linux dev
# environment for that reason - this file has only been import/syntax
# checked here, not exercised against (real or fake) hardware.

from __future__ import annotations

from typing import Any

import numpy as np

from qcodes.instrument_drivers.signal_hound.SignalHound_USB_SA124B import (
    SignalHoundUSBSA124B,
)


class SignalHoundSA124B(SignalHoundUSBSA124B):
    """QCoDeS-native Signal Hound SA124B, with the same `config` dict shape
    `legacy/drivers/signal_hound.py::SignalHoundSA` used:
        'frequency': float (Hz) - center frequency
        'span': float (Hz)
        'bandwidth': float (Hz) - mapped to resolution bandwidth (`rbw`);
            the native driver also has an independent video bandwidth
            (`vbw`, a smoothing window on top of rbw) that the old single
            'bandwidth' value didn't distinguish - left at its default
            unless set separately via `self.vbw(...)`.
        'reference_level': float (dBm)
        'reference_osc': 'EXT' or 'INT'
    Anything not specified is left untouched.

    NOTE - 'mode': the old driver's zero-span ('ZS') mode has no
    equivalent here - the native driver's `device_mode` is hardcoded to
    "sweeping" (frequency-domain) and isn't settable at all.
    `configure(config)` raises `NotImplementedError` if asked for
    `'mode': 'ZS'` (or anything but 'SA') rather than silently measuring
    in the wrong mode.

    NOTE - 'initiate'/manual triggering: the old driver's
    `fire_trigger()` + `ask_if_done()` + `get_sa_data()` sequence is
    replaced by a single call to `trace()`/`get_trace()` below, which
    performs a full synchronous sweep and returns - there is no separate
    trigger-then-wait step to configure, so `'initiate'` is accepted but
    has no effect.

    NOTE - reference oscillator: switching to external reference is
    one-way (Signal Hound's own restriction - see `external_reference`'s
    docstring) - closing and reopening the connection is required to go
    back to internal. `configure(config)` never calls
    `external_reference(False)` for this reason.
    """

    def __init__(
        self,
        name: str,
        dll_path: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, dll_path=dll_path, **kwargs)
        self._dll_path = dll_path or self.dll_path
        self._config: dict[str, Any] = {}

        if config:
            self.configure(config)

    def get_config_info(self) -> dict[str, Any]:
        """Same shape as `legacy.drivers.basic_instrument.BasicInstrument
        .get_config_info` - `visa_address` holds the DLL path here since
        this instrument has no VISA address - so this instrument can be
        dropped straight into `BaseMeasurement.instruments`."""
        return {
            "visa_address": self._dll_path,
            "nickname": self.name,
            "config": self._config,
        }

    def configure(self, config: dict[str, Any] | None = None) -> None:
        """With `config`: apply it (see class docstring for keys/gaps),
        then re-sync + refresh trace setpoints - matches this folder's
        `configure(config)` convention on the other instruments. With no
        `config` (as the native driver's own internal callers, e.g.
        `_get_power_at_freq`, use it): behaves exactly like the inherited
        zero-argument `SignalHoundUSBSA124B.configure()`.
        """
        if config:
            self._config.update(config)
            mode = config.get("mode", "SA")
            if mode not in ("SA", "sa"):
                raise NotImplementedError(
                    f"mode={mode!r} is not supported by the native qcodes "
                    "SignalHoundUSBSA124B driver - it only supports "
                    "frequency-domain sweeping ('SA')."
                )
            if "frequency" in config:
                self.frequency(config["frequency"])
            if "span" in config:
                self.span(config["span"])
            if "bandwidth" in config:
                self.rbw(config["bandwidth"])
            if "reference_level" in config:
                self.ref_lvl(config["reference_level"])
            if str(config.get("reference_osc", "")).upper() == "EXT":
                self.external_reference(True)
        super().configure()

    def get_trace(self) -> tuple[np.ndarray, np.ndarray]:
        """`(frequencies, power)` for one full sweep - the native-driver
        equivalent of `legacy/drivers/signal_hound.py
        ::SignalHoundSA.get_sa_trace`."""
        return self.frequency_axis(), self.trace()

    def safe_shutdown(self) -> None:
        """Called by the measurement harness on any error/abort. Aborts
        any running acquisition (`close()`, called separately right after,
        does the same, but this makes it available mid-run too)."""
        self.abort()
