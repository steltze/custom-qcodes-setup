# ---------------------------------------------------------------------------
# AnaPico APUASYN20-X (4 channels)
# ---------------------------------------------------------------------------
#
# Stopgap wrapper: all hardware I/O is delegated to the proven
# exopy_hqc_legacy-based `Anapico4` driver (legacy/drivers/anapico.py),
# held here as `self._legacy`. This class only adds a QCoDeS-compatible
# surface (Parameters, validators, snapshot()) on top of it. Sibling of
# `AnaPicoAPUASYN20` (single-channel) - see that file for the channel-less
# equivalent of every parameter/method below.
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

from legacy.drivers.anapico import Anapico4

_CHANNELS = (1, 2, 3, 4)


class AnaPicoAPUASYN20X(Instrument):
    """
    QCoDeS-compatible wrapper around the exopy-based `Anapico4` driver.

    4 channels, 8 kHz - 20 GHz each. Channel numbers below are 1-indexed
    (matching every other multi-channel instrument in this repo, e.g.
    `KeysightM8195A`'s `amplitude_1`/`amplitude_4`) - `self._legacy
    .channels` underneath is the 0-indexed list `Anapico4`/
    `AnapicoNChannels` actually expose, so channel `ch` here is
    `self._legacy.channels[ch - 1]`.
    """

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)

        # Anapico4.__init__ already applies `config` itself (frequencies/
        # powers/phases/outputs/reference_osc, each a 4-tuple - see
        # legacy/drivers/anapico.py::Anapico4.set_config), same
        # convention as everywhere else in this repo.
        self._legacy = Anapico4(address, name, config=config)

        for ch in _CHANNELS:
            self.add_parameter(
                f"frequency_{ch}",
                label=f"Ch{ch} frequency",
                unit="Hz",
                get_cmd=lambda ch=ch: self._legacy.channels[ch - 1].frequency,
                set_cmd=lambda v, ch=ch: setattr(
                    self._legacy.channels[ch - 1], "frequency", v
                ),
                vals=Numbers(min_value=0),
            )

            self.add_parameter(
                f"power_{ch}",
                label=f"Ch{ch} output power",
                unit="dBm",
                get_cmd=lambda ch=ch: self._legacy.channels[ch - 1].power,
                set_cmd=lambda v, ch=ch: setattr(
                    self._legacy.channels[ch - 1], "power", v
                ),
                # Confirmed on real hardware (single-channel unit, same
                # AnapicoChannel underneath): an out-of-range set is
                # silently ignored rather than rejected - the write has
                # no effect, and the legacy driver's write-then-verify
                # check reports it as the confusing "Instrument did not
                # set correctly the power". Bounding it here fails fast
                # instead. VERIFY this range is the same across all 4
                # channels on your unit, not just channel 1.
                vals=Numbers(min_value=-10.0, max_value=23.0),
            )

            self.add_parameter(
                f"phase_{ch}",
                label=f"Ch{ch} phase",
                unit="rad",
                get_cmd=lambda ch=ch: self._legacy.channels[ch - 1].phase,
                set_cmd=lambda v, ch=ch: setattr(
                    self._legacy.channels[ch - 1], "phase", v
                ),
                vals=Numbers(-np.pi, np.pi),
            )

            self.add_parameter(
                f"output_enabled_{ch}",
                label=f"Ch{ch} output enabled",
                get_cmd=lambda ch=ch: self._legacy.channels[ch - 1].output,
                set_cmd=lambda v, ch=ch: setattr(
                    self._legacy.channels[ch - 1], "output", "ON" if v else "OFF"
                ),
                vals=Bool(),
            )

        # Shared across all 4 channels, not per-channel - matches
        # `Anapico4`/`AnapicoNChannels`, where the reference oscillator is
        # one instrument-level setting.
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

    def which_outputs_enabled(self) -> dict[int, bool]:
        """`{channel: is_output_enabled}` for all 4 channels in one call -
        the "which outputs are active" question, answered in one shot
        instead of reading `output_enabled_1..4` separately."""
        return {ch: getattr(self, f"output_enabled_{ch}")() for ch in _CHANNELS}

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
        error/abort only. Turns RF output off on all 4 channels."""
        for ch in _CHANNELS:
            getattr(self, f"output_enabled_{ch}")(False)

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
