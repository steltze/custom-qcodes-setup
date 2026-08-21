"""Keysight M8195A - raw SCPI over pyvisa, no qcodes or pyarbtools
dependency.

65 GSa/s arbitrary waveform generator (AXIe module; SCPI via the Soft
Front Panel's server, or the module's own vxi11/hislip server).

Every SCPI command and the waveform-download/tiling logic below is
transcribed from the installed `pyarbtools.instruments.M8195A` /
`SignalGeneratorBase` (this codebase used to wrap that library directly -
see `instruments_old/awg_M8195A.py`), plus this lab's own
`find_waveform_k_nper`/`farey_fraction` sample-rate-matching helpers from
that same file. In particular:

  * `_check_wfm`'s "wraparound": an under-length/mis-granular waveform is
    *tiled* (repeated whole), NOT zero-padded, until it's a multiple of
    `gran` (256 samples) and at least `min_len` (1280 samples) long.
    Getting this backwards silently produces a waveform with the wrong
    period.
  * `download_wfm`'s `trace<ch>:def <segment>, <length>` declares the
    segment using the length of the *original*, pre-tiling array, while
    the binary data written into it is the tiled (generally longer)
    array - a quirk of the pyarbtools source, preserved here rather than
    "fixed" since deviating from a real, in-the-wild library's behavior
    without a hardware way to check which is actually correct is riskier
    than reproducing it exactly. It isn't exercised by the one call path
    this repo actually uses (`send_sine*` below), which always passes an
    already gran-multiple-length array - tiling is a no-op there.

VERIFY-BEFORE-TRUST: none of this has been run against real M8195A
hardware yet (unlike its `anapico_apuasyn20*.py` siblings in this
package) - only against the pyarbtools source and
`tests/verify_native_drivers.py`'s fake SCPI resource. Cross-check
against the M8195A programming guide before trusting it on real hardware,
especially the binary waveform path.

fir_scale / mem_mode / output_reference_source: these three are NOT in
stock pyarbtools - transcribed from a locally-patched fork of
pyarbtools' instruments.py (`fir_instruments.py` at the repo root) found
after the fact, which adds real FIR-filter-scale control for the M8195A
(the datasheet-documented way to get output amplitudes below the normal
75mV DAC floor - see `set_amplitude`, whose own 75mV/1V range check is
unchanged by this: `fir_scale` is an independent, additional output
scaling stage, not a replacement for it). Deliberately NOT carried over
from that same file: its `gran = 128` / `min_len = 128` (down from the
standard 256/1280) - that file's own comment flags those as "guessed
values", never confirmed against hardware or the datasheet, so this
driver keeps the standard 256/1280 pyarbtools values instead.

Connection: unlike `AWG_M8195A` (which builds its own VISA address from a
bare IP - `tcpip::{ip}::inst0::instr`), this driver takes a normal full
VISA resource string via `VisaDriver`, matching every other driver in
this package - e.g. `"TCPIP0::localhost::inst0::INSTR"`.

For the qcodes-compatible instrument, see
`instruments_native/KeysightM8195A.py`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import VisaDriver

GRANULARITY = 256
MIN_LENGTH = 1280
BIN_MULT = 127
CHANNELS = (1, 2, 3, 4)


def _wraparound_repeats(length: int, gran: int, min_len: int) -> int:
    """How many whole copies of a `length`-sample waveform are needed so
    the tiled result is a multiple of `gran` and at least `min_len`
    samples. Matches pyarbtools' `wraparound_calc`."""
    repeats = 1
    total = length
    while total % gran != 0 or total < min_len:
        total += length
        repeats += 1
    return repeats


def _farey_fraction(
    x: float, max_iter: int = 1000, end_tol: float = 1e-4
) -> tuple[int, int, int, int]:
    """Best rational approximation `a/b` (and its Farey neighbor `c/d`) to
    `x`, found by mediant search - used by `send_sine_force_keep_rate` to
    find an exact `(n_per, k)` pair at a *fixed* sample rate. Transcribed
    from `instruments_old/awg_M8195A.py::farey_fraction`."""
    a, b = math.floor(x), 1
    c, d = math.ceil(x), 1
    for _ in range(max_iter):
        new_nom, new_denom = a + c, b + d
        new_approx = new_nom / new_denom
        if new_approx <= x:
            a, b = new_nom, new_denom
        else:
            c, d = new_nom, new_denom
        if abs(new_approx - x) <= end_tol:
            if abs(a / b - x) <= abs(c / d - x):
                return a, b, c, d
            return c, d, a, b
    raise RecursionError(
        f"Could not find a good enough rational approximation to {x}; "
        "increase max_iter."
    )


class KeysightM8195A(VisaDriver):
    """Keysight M8195A 65 GSa/s AWG."""

    default_terminator = "\n"
    min_rate = 53.76e9
    max_rate = 65e9
    gran = GRANULARITY
    min_len = MIN_LENGTH

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        reset: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, address, **kwargs)
        self._config: dict[str, Any] = {}

        if reset:
            self.write("*rst")
            self.ask("*opc?")
            self.write("abort")

        if config:
            self.configure(config)

        self.connect_message()

    # -- basic configuration ------------------------------------------------
    def get_dac_mode(self) -> str:
        return self.ask("inst:dacm?").strip().lower()

    def set_dac_mode(self, value: str) -> None:
        if value not in ("single", "dual", "four", "marker", "dcd", "dcmarker"):
            raise ValueError(
                "dac_mode must be one of 'single', 'dual', 'four', "
                "'marker', 'dcd', 'dcmarker'"
            )
        self.write(f"inst:dacm {value}")

    def get_mem_div(self) -> int:
        raw = self.ask("instrument:memory:extended:rdivider?").strip()
        return int(raw.split("DIV")[-1])

    def set_mem_div(self, value: int) -> None:
        if value not in (1, 2, 4):
            raise ValueError("mem_div must be 1, 2, or 4")
        self.write(f"instrument:memory:extended:rdivider div{value}")

    def get_sample_rate(self) -> float:
        """Sa/s."""
        return float(self.ask("frequency:raster?").strip())

    def set_sample_rate(self, value: float) -> None:
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        self.write(f"frequency:raster {value}")

    def get_function_mode(self) -> str:
        return self.ask("func:mode?").strip()

    def set_function_mode(self, value: str) -> None:
        if value.lower() not in ("arb", "sts", "stsc"):
            raise ValueError("function_mode must be 'arb', 'sts', or 'stsc'")
        self.write(f"func:mode {value}")

    def get_reference_source(self) -> str:
        return self.ask("roscillator:source?").strip()

    def set_reference_source(self, value: str) -> None:
        if value.lower() not in ("axi", "int", "ext"):
            raise ValueError("reference_source must be 'axi', 'int', or 'ext'")
        self.write(f"roscillator:source {value}")

    def get_reference_frequency(self) -> float:
        """Hz."""
        return float(self.ask("roscillator:frequency?").strip())

    def set_reference_frequency(self, value: float) -> None:
        if value <= 0:
            raise ValueError("reference_frequency must be positive")
        self.write(f"roscillator:frequency {value}")

    def get_amplitude(self, channel: int) -> float:
        """Peak-to-peak output amplitude, V."""
        return float(self.ask(f"voltage{channel}?"))

    def set_amplitude(self, channel: int, value: float) -> None:
        if channel not in CHANNELS:
            raise ValueError("channel must be 1, 2, 3, or 4")
        if not (0.075 <= value <= 1.0):
            raise ValueError("amplitude must be between 75 mV and 1 V")
        self.write(f"voltage{channel} {value}")

    def get_output_enabled(self, channel: int) -> bool:
        return self.ask(f"OUTPUT{channel}?").strip() in ("1", "ON")

    def set_output_enabled(self, channel: int, value: bool) -> None:
        self.write(f"OUTPUT{channel} {'ON' if value else 'OFF'}")

    # -- FIR filter scale / memory mode / output reference source -------------
    # Not in stock pyarbtools - see module docstring (transcribed from
    # fir_instruments.py).
    _FIR_RATE_NAME = {1: "frat", 2: "hrat", 4: "qrat"}

    def get_fir_scale(self, channel: int) -> float:
        """FIR filter output scale for `channel`, 0-1 - the
        datasheet-documented way to reach output amplitudes below the
        normal 75mV `amplitude` floor. Independent of `amplitude`: this
        scales the signal *after* the amplitude stage, it doesn't replace
        it - `amplitude` itself is still clamped to [75mV, 1V]."""
        rate = self._FIR_RATE_NAME[self.get_mem_div()]
        return float(self.ask(f"outp{channel}:filt:{rate}:scal?"))

    def set_fir_scale(self, channel: int, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("fir_scale must be between 0 and 1")
        rate = self._FIR_RATE_NAME[self.get_mem_div()]
        self.write(f"outp{channel}:filt:{rate}:scal {value}")

    def get_mem_mode(self, channel: int) -> str:
        """'INT' or 'EXT'."""
        return self.ask(f"trac{channel}:mmod?").strip()

    def set_mem_mode(self, channel: int, value: str) -> None:
        if value.lower() not in ("int", "ext"):
            raise ValueError("mem_mode must be 'int' or 'ext'")
        self.write(f"trac{channel}:mmod {value}")

    def get_output_reference_source(self) -> str:
        """'INT', 'EXT', 'SCLK1', or 'SCLK2' - the AWG's reference-clock
        *output* routing, not its own timebase source (see
        `reference_source`/`roscillator:source` for that)."""
        return self.ask("outp:rosc:source?").strip()

    def set_output_reference_source(self, value: str) -> None:
        if value.lower() not in ("int", "ext", "sclk1", "sclk2"):
            raise ValueError(
                "output_reference_source must be 'int', 'ext', 'sclk1', or 'sclk2'"
            )
        self.write(f"outp:rosc:source {value}")

    def configure(self, config: dict[str, Any]) -> None:
        """Apply a config dict - same keys as pyarbtools' `configure()`,
        plus this driver's FIR/memory-mode/output-reference additions:
            'dac_mode', 'mem_div', 'sample_rate', 'reference_source',
            'reference_frequency', 'function_mode', 'output_reference_source',
            'amplitude_1'..'amplitude_4', 'fir_scale_1'..'fir_scale_4',
            'mem_mode_1'..'mem_mode_4'
        Stops output on all 4 channels first, matching pyarbtools'
        `configure()` - a fresh config shouldn't leave a stale waveform
        playing on a channel it didn't touch."""
        self._config.update(config)
        for ch in CHANNELS:
            self.stop(ch)
        if "dac_mode" in config:
            self.set_dac_mode(config["dac_mode"])
        if "mem_div" in config:
            self.set_mem_div(config["mem_div"])
        if "sample_rate" in config:
            self.set_sample_rate(config["sample_rate"])
        if "reference_source" in config:
            self.set_reference_source(config["reference_source"])
        if "reference_frequency" in config:
            self.set_reference_frequency(config["reference_frequency"])
        if "output_reference_source" in config:
            self.set_output_reference_source(config["output_reference_source"])
        if "function_mode" in config:
            self.set_function_mode(config["function_mode"])
        for ch in CHANNELS:
            key = f"amplitude_{ch}"
            if key in config:
                self.set_amplitude(ch, config[key])
            key = f"mem_mode_{ch}"
            if key in config:
                self.set_mem_mode(ch, config[key])
            key = f"fir_scale_{ch}"
            if key in config:
                self.set_fir_scale(ch, config[key])
        self.check_error()

    def check_error(self) -> None:
        """Print and raise on anything in the error queue. Matches
        pyarbtools' `SignalGeneratorBase.err_check` (called at the end of
        its own `configure()`), including its exact "no error" comparison
        (all `+`/`-` characters stripped before comparing)."""
        errors = []
        while True:
            # No leading colon - matches pyarbtools' literal 'SYST:ERR?',
            # unlike the Anapico siblings' `:SYST:ERR?` (VisaDriver's
            # default `get_error()` command assumes the latter dialect).
            raw = self.get_error("SYST:ERR?").strip().replace("+", "").replace("-", "")
            if raw == '0,"No error"':
                break
            print(raw)
            errors.append(raw)
        if errors:
            raise RuntimeError(errors)

    # -- run control ----------------------------------------------------------
    def play(self, segment: int, channel: int = 1) -> None:
        """Select waveform `segment`, turn on `channel`'s analog output,
        and begin continuous playback. Needed after every `configure()`
        call (e.g. an amplitude change) - `configure()` stops output on
        every channel first, so playback of an already-downloaded segment
        has to be explicitly resumed."""
        self.write(f"trace:select {segment}")
        self.write(f"output{channel} on")
        self.write("init:cont on")
        self.write("init:imm")

    def stop(self, channel: int = 1) -> None:
        """Turn off `channel`'s analog output and abort playback."""
        self.write(f"output{channel} off")
        self.write("abort")

    def delete_segment(self, segment: int, channel: int = 1) -> None:
        """Delete one waveform segment from a channel's segment memory."""
        self.write("abort")
        self.write(f"trace{channel}:del {segment}")

    def clear_all_wfm(self) -> None:
        """Clear all segments from segment memory, on every channel."""
        self.write("abort")
        for ch in CHANNELS:
            self.write(f"trace{ch}:del:all")

    def ask_if_done(self) -> str:
        """Block until the AWG has finished processing pending commands
        (e.g. the `play()` issued by `send_sine`)."""
        return self.ask("*OPC?")

    # -- waveform handling --------------------------------------------------
    def _check_wfm(self, wfm_data: np.ndarray) -> np.ndarray:
        """Tile (repeat whole, not zero-pad) `wfm_data` until it's a
        multiple of `gran` and at least `min_len` samples, then scale to
        the AWG's signed-8-bit DAC range. Matches pyarbtools'
        `check_wfm`."""
        repeats = _wraparound_repeats(len(wfm_data), self.gran, self.min_len)
        if repeats > 1:
            print(f"Information: Waveform repeated {repeats} times.")
        wfm = np.tile(wfm_data, repeats)
        if len(wfm) < self.min_len:
            raise ValueError(
                f"Waveform length {len(wfm)} must be at least {self.min_len}."
            )
        if len(wfm) % self.gran != 0:
            raise ValueError(f"Waveform must have a granularity of {self.gran}.")
        return np.array(BIN_MULT * wfm, dtype=np.int8)

    def download_wfm(self, wfm_data: np.ndarray, channel: int = 1, name: str = "wfm") -> int:
        """Download a waveform into `channel`'s segment memory; returns
        the new segment number to pass to `play()`. `wfm_data` should be
        float in [-1, 1]. Matches pyarbtools' `download_wfm` (including
        its `trace<ch>:def` quirk - see module docstring)."""
        self.write("abort")
        wfm = self._check_wfm(wfm_data)
        length = len(wfm_data)
        segment = int(self.ask(f"trace{channel}:catalog?").strip().split(",")[-2]) + 1
        self.write(f"trace{channel}:def {segment}, {length}")
        self.resource.write_binary_values(
            f"trace{channel}:data {segment}, 0, ", wfm, datatype="b"
        )
        self.write(f'trace{channel}:name {segment},"{name}_{segment}"')
        return segment

    def find_waveform_k_nper(
        self, freq: float, max_iterations: int = 1000
    ) -> tuple[int, int, float]:
        """Find `(k, n_per, rate)` such that a sine of `freq` fits in `k`
        AWG granularities of samples over `n_per` periods, at a sample
        `rate` within `[min_rate, max_rate]`. Transcribed from
        `instruments_old/awg_M8195A.py::AWG_M8195A.find_waveform_k_nper`."""
        k = 1
        n_per = 1
        for _ in range(max_iterations):
            current_rate = 1000 * round(self.gran * freq * k / n_per / 1000)
            if current_rate >= self.max_rate:
                n_per += 1
            elif current_rate <= self.min_rate:
                k += 1
                n_per = k + 1
            else:
                return k, n_per, current_rate
        raise RuntimeError(f"Couldn't find k and n_per for freq {freq}Hz.")

    def send_sine(
        self, freq: float, phase: float, channel: int, amp: float | None = None
    ) -> int:
        """Recompute the sample rate for clean granularity, then download
        and play a sine of `freq` Hz / `phase` rad on `channel`. Changes
        the AWG's sample rate - see `send_sine_keep_rate`/
        `send_sine_force_keep_rate` if that's not acceptable."""
        k, n_per, rate = self.find_waveform_k_nper(freq)
        t_points = np.arange(k * self.gran)
        period = k * self.gran / n_per
        amp_points = np.sin(2 * np.pi * t_points / period + phase)
        self.set_sample_rate(rate)
        if amp is not None:
            self.set_amplitude(channel, amp)
        segment = self.download_wfm(amp_points, channel=channel, name="wfm")
        self.play(segment, channel=channel)
        return segment

    def send_sine_keep_rate(
        self, freq: float, phase: float, channel: int, amp: float | None = None
    ) -> int:
        """Like `send_sine`, but raises if `freq` isn't compatible with
        the current sample rate instead of changing it."""
        k, n_per, rate = self.find_waveform_k_nper(freq)
        current_rate = self.get_sample_rate()
        if rate != current_rate:
            raise RuntimeError(
                "Tried to send an additional signal incompatible with the "
                f"current rate. Current {current_rate}Hz, needed {rate}Hz."
            )
        t_points = np.arange(k * self.gran)
        period = k * self.gran / n_per
        amp_points = np.sin(2 * np.pi * t_points / period + phase)
        if amp is not None:
            self.set_amplitude(channel, amp)
        segment = self.download_wfm(amp_points, channel=channel, name="wfm")
        self.play(segment, channel=channel)
        return segment

    def send_sine_force_keep_rate(
        self, freq: float, phase: float, channel: int, amp: float | None = None
    ) -> int:
        """Like `send_sine`, but keeps the current sample rate fixed and
        instead lengthens the waveform (Farey-fraction approximation) to
        fit `freq` as closely as possible."""
        decimal_val = self.gran * freq / self.get_sample_rate()
        n_per, k, _, _ = _farey_fraction(decimal_val, max_iter=1999, end_tol=1e-6)
        t_points = np.arange(k * self.gran)
        period = k * self.gran / n_per
        amp_points = np.sin(2 * np.pi * t_points / period + phase)
        if amp is not None:
            self.set_amplitude(channel, amp)
        segment = self.download_wfm(amp_points, channel=channel, name="wfm")
        self.play(segment, channel=channel)
        return segment

    def safe_shutdown(self) -> None:
        """Called by the measurement harness's `safe_run()` on *every*
        exit from a run - a clean finish as much as an error/abort, not
        error/abort only. Turns output off on channels 1 and 4 (the pair
        used in 'dual' DAC mode, this repo's default) rather than wiping
        segment memory, so it's fast and doesn't lose the
        last-programmed waveform."""
        self.stop(1)
        self.stop(4)
