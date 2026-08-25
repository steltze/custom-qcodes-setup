# ---------------------------------------------------------------------------
# Keysight M8195A
# ---------------------------------------------------------------------------
#
# Stopgap wrapper: all hardware I/O is delegated to the proven pyarbtools-
# based `AWG_M8195A` driver (legacy/drivers/awg_M8195A.py), held here as
# `self._legacy`. This class only adds a QCoDeS-compatible surface
# (Parameters, validators, snapshot()) on top of it - in particular it does
# NOT redo pyarbtools' waveform granularity/padding logic.
#
# `AWG_M8195A` now imports its `M8195A` base from `fir_instruments.py`
# (a locally-patched pyarbtools fork), not stock pyarbtools - see that
# import's comment in awg_M8195A.py. That's what makes `fir_scale_N`/
# `mem_mode_N` below possible; without it `configure(fir_scale1=...)`
# would raise KeyError, same as native.drivers.keysight_m8195a.py's
# module docstring found true for every real pyarbtools release.
#
# To detach from pyarbtools later: pick one parameter, replace its
# get_cmd/set_cmd with raw SCPI (verified against the M8195A programming
# guide), and move on to the next. Nothing outside this file needs to
# change while you do that.

from __future__ import annotations

from typing import Any

from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.validators import Bool, Enum, Numbers

from legacy.drivers.awg_M8195A import AWG_M8195A

class KeysightM8195A(Instrument):
    """
    QCoDeS-compatible wrapper around the pyarbtools-based `AWG_M8195A`.

    Every parameter below reads/writes an attribute of the wrapped
    `AWG_M8195A` instance via its `configure()` method (e.g. `sample_rate`
    <-> `self._legacy.fs` / `configure(fs=...)`). Waveform upload/playback
    is forwarded straight to pyarbtools' `download_wfm`/`play`.
    """

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)

        self._legacy = AWG_M8195A(address, name, config=config)

        self.dac_mode: Parameter = self.add_parameter(
            "dac_mode",
            label="DAC mode",
            get_cmd=lambda: self._legacy.dacMode,
            set_cmd=lambda v: self._legacy.configure(dacMode=v),
            vals=Enum("single", "dual", "four", "marker", "dcd", "dcmarker"),
        )

        self.mem_div: Parameter = self.add_parameter(
            "mem_div",
            label="Memory divider",
            get_cmd=lambda: self._legacy.memDiv,
            set_cmd=lambda v: self._legacy.configure(memDiv=v),
            vals=Enum(1, 2, 4),
        )

        self.sample_rate: Parameter = self.add_parameter(
            "sample_rate",
            label="Sample rate",
            unit="Sa/s",
            get_cmd=lambda: self._legacy.fs,
            set_cmd=lambda v: self._legacy.configure(fs=v),
            vals=Numbers(self._legacy.min_rate, self._legacy.max_rate),
        )

        self.reference_source: Parameter = self.add_parameter(
            "reference_source",
            label="Reference clock source",
            get_cmd=lambda: self._legacy.refSrc,
            set_cmd=lambda v: self._legacy.configure(refSrc=v),
            vals=Enum("axi", "int", "ext"),
        )

        self.reference_frequency: Parameter = self.add_parameter(
            "reference_frequency",
            label="Reference clock frequency",
            unit="Hz",
            get_cmd=lambda: self._legacy.refFreq,
            set_cmd=lambda v: self._legacy.configure(refFreq=v),
            vals=Numbers(min_value=0),
        )

        self.function_mode: Parameter = self.add_parameter(
            "function_mode",
            label="Function mode",
            get_cmd=lambda: self._legacy.func,
            set_cmd=lambda v: self._legacy.configure(func=v),
            vals=Enum("arb", "sts", "stsc"),
        )

        # Only channels 1 and 4 are set by AWG_M8195A.default_config (the
        # pair used in 'dual' DAC mode); the underlying driver itself
        # supports all four - add amplitude_2/3 (and fir_scale_2/3/
        # mem_mode_2/3) the same way if you switch to 'four' mode.
        for ch in (1, 4):
            self.add_parameter(
                f"amplitude_{ch}",
                label=f"Ch{ch} amplitude",
                unit="V",
                get_cmd=lambda ch=ch: getattr(self._legacy, f"amp{ch}"),
                set_cmd=lambda v, ch=ch: self._legacy.configure(**{f"amp{ch}": v}),
                vals=Numbers(0.075, 1.0),
            )

            # FIR filter output scale, 0-1 - the datasheet-documented way
            # to reach output amplitudes below the 75mV `amplitude_N`
            # floor. Independent of amplitude_N: scales the signal after
            # the amplitude stage, doesn't replace its own 75mV/1V range
            # check. See native/drivers/keysight_m8195a.py's module
            # docstring for the full story on where this came from.
            self.add_parameter(
                f"fir_scale_{ch}",
                label=f"Ch{ch} FIR filter scale",
                get_cmd=lambda ch=ch: getattr(self._legacy, f"fir_scale{ch}"),
                set_cmd=lambda v, ch=ch: self._legacy.configure(**{f"fir_scale{ch}": v}),
                vals=Numbers(0.0, 1.0),
            )

            self.add_parameter(
                f"mem_mode_{ch}",
                label=f"Ch{ch} memory mode",
                get_cmd=lambda ch=ch: getattr(self._legacy, f"mem_mode{ch}"),
                set_cmd=lambda v, ch=ch: self._legacy.configure(**{f"mem_mode{ch}": v}),
                vals=Enum("int", "ext", "INT", "EXT"),
            )

            # pyarbtools' play()/stop() (below) are write-only - they issue
            # `OUTPUT<ch> ON/OFF` but never read it back, so there was no
            # way to check a channel's actual output state. This queries
            # the same thing directly, bypassing pyarbtools (it has no
            # getter for this at all) via self._legacy's raw query/write -
            # same pattern as ask_if_done() below. `OUTPUT<ch>?` per the
            # M8195A/M8190A SCPI command set - VERIFY against real
            # hardware if this doesn't match (the write form is already
            # confirmed working, since play()/stop() use the identical
            # `OUTPUT<ch> ON/OFF` write).
            self.add_parameter(
                f"output_enabled_{ch}",
                label=f"Ch{ch} output enabled",
                get_cmd=lambda ch=ch: self._legacy.query(f"OUTPUT{ch}?").strip() in ("1", "ON"),
                set_cmd=lambda v, ch=ch: self._legacy.write(f"OUTPUT{ch} {'ON' if v else 'OFF'}"),
                vals=Bool(),
            )

    def get_idn(self) -> dict[str, str | None]:
        """`self._legacy` (pyarbtools) already queried `*idn?` once at
        connect time and cached it as `.instId` - reuse that instead of
        the base `Instrument.get_idn`, which calls `self.ask(...)` and
        plain `Instrument` (unlike `VisaInstrument`) doesn't implement."""
        idparts = [p.strip() for p in self._legacy.instId.split(",", 3)]
        idparts += [None] * (4 - len(idparts))
        return dict(zip(("vendor", "model", "serial", "firmware"), idparts))

    def get_config_info(self) -> dict[str, Any]:
        """Same shape as `legacy.drivers.basic_instrument.BasicInstrument
        .get_config_info` - `self._legacy` already implements it (it *is*
        a `BasicInstrument`), just forward to it so this instrument can be
        dropped straight into `BaseMeasurement.instruments`."""
        return self._legacy.get_config_info()

    # -- waveform / run control, forwarded straight to pyarbtools ----------
    def send_sine(
        self, freq: float, phase: float, channel: int, amp: float | None = None
    ) -> int:
        """Recalculate the sample rate for clean granularity, then
        download and play a sine on `channel`. See `AWG_M8195A.send_sine`."""
        return self._legacy.send_sine(freq, phase, channel, amp=amp)

    def send_sine_keep_rate(
        self, freq: float, phase: float, channel: int, amp: float | None = None
    ) -> int:
        """Like `send_sine`, but raises if `freq` isn't compatible with the
        current sample rate. See `AWG_M8195A.send_sine_keep_rate`."""
        return self._legacy.send_sine_keep_rate(freq, phase, channel, amp=amp)

    def send_sine_force_keep_rate(
        self, freq: float, phase: float, channel: int, amp: float | None = None
    ) -> int:
        """Like `send_sine`, but keeps the current sample rate by
        lengthening the waveform (Farey approximation). See
        `AWG_M8195A.send_sine_force_keep_rate`."""
        return self._legacy.send_sine_force_keep_rate(freq, phase, channel, amp=amp)

    def play(self, seg_id: int, ch: int = 1) -> None:
        """Select waveform `seg_id`, turn on channel `ch`'s analog output,
        and begin continuous playback. Needed after every `configure()`
        call (e.g. an amplitude change) - `pyarbtools`'s `configure()`
        stops output on every channel first, so playback of an
        already-downloaded segment has to be explicitly resumed."""
        self._legacy.play(seg_id, ch=ch)

    def delete_segment(self, seg_id: int, ch: int = 1) -> None:
        """Delete one waveform segment from a channel's segment memory."""
        self._legacy.delete_segment(seg_id, ch=ch)

    def clear_all_wfm(self) -> None:
        """Clear all segments from segment memory, on every channel."""
        self._legacy.clear_all_wfm()

    def stop(self, ch: int = 1) -> None:
        """Turn off one channel's analog output and abort playback."""
        self._legacy.stop(ch=ch)

    def safe_shutdown(self) -> None:
        """Called by the measurement harness's `safe_run()` on *every*
        exit from a run - a clean finish as much as an error/abort, not
        error/abort only. Turns output off on both channels used in
        'dual' DAC mode (1 and 4 - see the `default_config` note above)
        rather than wiping segment memory, so it's fast and doesn't lose
        the last-programmed waveform."""
        self.stop(1)
        self.stop(4)

    def ask_if_done(self) -> str:
        """Block until the AWG has finished processing pending commands
        (e.g. the `play()` issued by `send_sine`)."""
        return self._legacy.query("*OPC?")
