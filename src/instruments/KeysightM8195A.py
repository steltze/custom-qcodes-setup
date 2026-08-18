# ---------------------------------------------------------------------------
# Keysight M8195A
# ---------------------------------------------------------------------------
#
# Stopgap wrapper: all hardware I/O is delegated to the proven pyarbtools-
# based `AWG_M8195A` driver (instruments_old/awg_M8195A.py), held here as
# `self._legacy`. This class only adds a QCoDeS-compatible surface
# (Parameters, validators, snapshot()) on top of it - in particular it does
# NOT redo pyarbtools' waveform granularity/padding logic.
#
# To detach from pyarbtools later: pick one parameter, replace its
# get_cmd/set_cmd with raw SCPI (verified against the M8195A programming
# guide), and move on to the next. Nothing outside this file needs to
# change while you do that.

from __future__ import annotations

from typing import Any

from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.validators import Enum, Numbers

from instruments_old.awg_M8195A import AWG_M8195A

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

        # vals below reflect pyarbtools.instruments.M8195A's own validation
        # (checked against pyarbtools 2025.06.1) - configure() re-raises
        # ValueError itself for anything outside these sets, this just
        # fails fast with a clearer message.
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
        # pair used in 'dual' DAC mode); pyarbtools itself supports all
        # four - add amplitude_2/3 the same way if you switch to 'four'
        # mode. There is no mem_mode1..4 in pyarbtools' configure(); if you
        # need it, it'll have to be raw SCPI.
        for ch in (1, 4):
            self.add_parameter(
                f"amplitude_{ch}",
                label=f"Ch{ch} amplitude",
                unit="V",
                get_cmd=lambda ch=ch: getattr(self._legacy, f"amp{ch}"),
                set_cmd=lambda v, ch=ch: self._legacy.configure(**{f"amp{ch}": v}),
                vals=Numbers(0.075, 1.0),
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
        """Same shape as `instruments_old.basic_instrument.BasicInstrument
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
        """Called by the measurement harness on any error/abort. Turns
        output off on both channels used in 'dual' DAC mode (1 and 4 - see
        the `default_config` note above) rather than wiping segment
        memory, so it's fast and doesn't lose the last-programmed
        waveform."""
        self.stop(1)
        self.stop(4)

    def ask_if_done(self) -> str:
        """Block until the AWG has finished processing pending commands
        (e.g. the `play()` issued by `send_sine`).

        NOTE: this method does not exist on `AWG_M8195A`/pyarbtools (checked
        against the installed pyarbtools 2025.6.1 and this repo's
        `instruments_old/awg_M8195A.py`) even though the old
        `spectro_awgPump_sweep_...py` script calls it - that call was
        already broken in the checked-in script. Implemented here as a
        standard SCPI `*OPC?` blocking query, the same idiom
        `agilent_pna.py::run_averaging` uses to wait for the PNA."""
        return self._legacy.query("*OPC?")
