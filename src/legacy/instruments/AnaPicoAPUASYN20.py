# ---------------------------------------------------------------------------
# AnaPico APUASYN20
# ---------------------------------------------------------------------------
#
# Stopgap wrapper: all hardware I/O is delegated to the proven
# exopy_hqc_legacy-based `Anapico1` driver (legacy/drivers/anapico.py),
# held here as `self._legacy`. This class only adds a QCoDeS-compatible
# surface (Parameters, validators, snapshot()) on top of it.
#
# To detach from exopy later: pick one parameter, replace its
# get_cmd/set_cmd with raw SCPI - the exact command strings are already
# visible in legacy/drivers/exopy_hqc_legacy/drivers/visa/anapico.py, so
# it's mostly transcription once you've verified each one against the
# manual - and move on to the next. Nothing outside this file needs to
# change while you do that.

from __future__ import annotations

from typing import Any

import numpy as np

from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.validators import Bool, Enum, Numbers

from legacy.drivers.anapico import Anapico1


class AnaPicoAPUASYN20(Instrument):
    """
    QCoDeS-compatible wrapper around the exopy-based `Anapico1` driver.

    Single-channel, 8 kHz - 20 GHz. For the multi-channel APUASYN20-X,
    wrap `Anapico4` (or `AnapicoNChannels` with the right `n_channels`)
    the same way instead.
    """

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)

        self._legacy = Anapico1(address, name, config=config)
        self._chan = self._legacy.channels[0]

        self.frequency: Parameter = self.add_parameter(
            "frequency",
            label="Frequency",
            unit="Hz",
            get_cmd=lambda: self._chan.frequency,
            set_cmd=lambda v: setattr(self._chan, "frequency", v),
            vals=Numbers(min_value=0),
        )

        self.power: Parameter = self.add_parameter(
            "power",
            label="Output power",
            unit="dBm",
            get_cmd=lambda: self._chan.power,
            set_cmd=lambda v: setattr(self._chan, "power", v),
            # Confirmed on real hardware: an out-of-range set (e.g. -20
            # dBm) isn't rejected with a clear error, it's silently
            # ignored - the write has no effect and the channel is left
            # at whatever it was, which the legacy driver's write-then-
            # verify check then reports as the confusing
            # "Instrument did not set correctly the power". Bounding it
            # here fails fast instead.
            vals=Numbers(min_value=-10.0, max_value=23.0),
        )

        self.phase: Parameter = self.add_parameter(
            "phase",
            label="Phase",
            unit="rad",
            get_cmd=lambda: self._chan.phase,
            set_cmd=lambda v: setattr(self._chan, "phase", v),
            vals=Numbers(-np.pi, np.pi),
        )

        self.output_enabled: Parameter = self.add_parameter(
            "output_enabled",
            label="Output enabled",
            get_cmd=lambda: self._chan.output,
            set_cmd=lambda v: setattr(self._chan, "output", "ON" if v else "OFF"),
            vals=Bool(),
        )

        self.reference_source: Parameter = self.add_parameter(
            "reference_source",
            label="Reference oscillator source",
            get_cmd=lambda: self._legacy.ref_oscillator,
            set_cmd=lambda v: setattr(self._legacy, "ref_oscillator", v),
            vals=Enum("INT", "EXT"),
        )

        self.oscillator_locked: Parameter = self.add_parameter(
            "oscillator_locked",
            label="Reference oscillator locked",
            get_cmd=lambda: self._legacy.oscillator_lock,
            vals=Bool(),
        )

    def get_idn(self) -> dict[str, str | None]:
        """Plain `Instrument` (unlike `VisaInstrument`) doesn't implement
        `self.ask(...)`, which the base `Instrument.get_idn` relies on -
        query through the wrapped exopy driver's own connection instead."""
        idparts = [p.strip() for p in self._legacy.query("*IDN?").split(",", 3)]
        idparts += [None] * (4 - len(idparts))
        return dict(zip(("vendor", "model", "serial", "firmware"), idparts))

    def get_config_info(self) -> dict[str, Any]:
        """Same shape as `legacy.drivers.basic_instrument.BasicInstrument
        .get_config_info` - `self._legacy` already implements it (it *is*
        a `BasicInstrument`), just forward to it so this instrument can be
        dropped straight into `BaseMeasurement.instruments`."""
        return self._legacy.get_config_info()

    def safe_shutdown(self) -> None:
        """Called by the measurement harness's `safe_run()` on *every*
        exit from a run - a clean finish as much as an error/abort, not
        error/abort only. Turns RF output off."""
        self.output_enabled(False)

    # -- diagnostics, not covered by the exopy driver -----------------------
    def get_error(self) -> str:
        """Pop one entry off the instrument's error queue."""
        return self._legacy.query(":SYST:ERR?")

    def flush_errors(self) -> list[str]:
        """Drain the error queue; returns everything that was queued."""
        errors = []
        for _ in range(50):
            err = self.get_error()
            if err.startswith(("0,", "+0,")):
                break
            errors.append(err)
        return errors
