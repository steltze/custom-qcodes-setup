"""Keysight M8195A - qcodes-compatible instrument.

One instrument class, built via multiple inheritance directly on top of
`drivers.keysight_m8195a.KeysightM8195A` - see
`instruments_native/AnaPicoAPUASYN20.py`'s module docstring for why (base
order, the `name`/`write`/`ask` collisions with plain
`qcodes.instrument.Instrument`, the explicit `close()` override, and why
the driver exposes plain `get_x()`/`set_x()` methods rather than
descriptor-based properties).

Waveform methods (`send_sine`, `download_wfm`, `play`, `stop`, ...) are
inherited directly from the driver, unchanged - they already don't touch
qcodes at all.

`fir_scale_1`..`fir_scale_4` are the datasheet-documented way to reach
output amplitudes below the normal 75mV `amplitude_N` floor - see the
driver's module docstring for where this (and `mem_mode_N`/
`output_reference_source`) came from.
"""

from __future__ import annotations

from typing import Any

from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.validators import Bool, Enum, Numbers

from drivers.keysight_m8195a import CHANNELS
from drivers.keysight_m8195a import KeysightM8195A as _RawM8195A


class KeysightM8195A(_RawM8195A, Instrument):
    """Keysight M8195A 65 GSa/s AWG, as a qcodes instrument. See
    `drivers.keysight_m8195a.KeysightM8195A` for the underlying SCPI and
    waveform-download logic, and its VERIFY-BEFORE-TRUST note - none of
    this has been run against real M8195A hardware yet."""

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        reset: bool = True,
        **kwargs: Any,
    ) -> None:
        Instrument.__init__(self, name, **kwargs)
        _RawM8195A.__init__(self, name, address, config=config, reset=reset)

        self.dac_mode: Parameter = self.add_parameter(
            "dac_mode",
            label="DAC mode",
            get_cmd=self.get_dac_mode,
            set_cmd=self.set_dac_mode,
            vals=Enum("single", "dual", "four", "marker", "dcd", "dcmarker"),
        )

        self.mem_div: Parameter = self.add_parameter(
            "mem_div",
            label="Memory divider",
            get_cmd=self.get_mem_div,
            set_cmd=self.set_mem_div,
            vals=Enum(1, 2, 4),
        )

        self.sample_rate: Parameter = self.add_parameter(
            "sample_rate",
            label="Sample rate",
            unit="Sa/s",
            get_cmd=self.get_sample_rate,
            set_cmd=self.set_sample_rate,
            vals=Numbers(self.min_rate, self.max_rate),
        )

        self.reference_source: Parameter = self.add_parameter(
            "reference_source",
            label="Reference clock source",
            get_cmd=self.get_reference_source,
            set_cmd=self.set_reference_source,
            vals=Enum("axi", "int", "ext"),
        )

        self.reference_frequency: Parameter = self.add_parameter(
            "reference_frequency",
            label="Reference clock frequency",
            unit="Hz",
            get_cmd=self.get_reference_frequency,
            set_cmd=self.set_reference_frequency,
            vals=Numbers(min_value=0),
        )

        self.function_mode: Parameter = self.add_parameter(
            "function_mode",
            label="Function mode",
            get_cmd=self.get_function_mode,
            set_cmd=self.set_function_mode,
            vals=Enum("arb", "sts", "stsc"),
        )

        self.output_reference_source: Parameter = self.add_parameter(
            "output_reference_source",
            label="Reference clock output routing",
            get_cmd=self.get_output_reference_source,
            set_cmd=self.set_output_reference_source,
            vals=Enum("int", "ext", "sclk1", "sclk2"),
        )

        for ch in CHANNELS:
            self.add_parameter(
                f"amplitude_{ch}",
                label=f"Ch{ch} amplitude",
                unit="V",
                get_cmd=(lambda ch=ch: self.get_amplitude(ch)),
                set_cmd=(lambda v, ch=ch: self.set_amplitude(ch, v)),
                vals=Numbers(0.075, 1.0),
            )
            self.add_parameter(
                f"output_enabled_{ch}",
                label=f"Ch{ch} output enabled",
                get_cmd=(lambda ch=ch: self.get_output_enabled(ch)),
                set_cmd=(lambda v, ch=ch: self.set_output_enabled(ch, v)),
                vals=Bool(),
            )
            self.add_parameter(
                f"fir_scale_{ch}",
                label=f"Ch{ch} FIR filter scale",
                get_cmd=(lambda ch=ch: self.get_fir_scale(ch)),
                set_cmd=(lambda v, ch=ch: self.set_fir_scale(ch, v)),
                vals=Numbers(0.0, 1.0),
            )
            self.add_parameter(
                f"mem_mode_{ch}",
                label=f"Ch{ch} memory mode",
                get_cmd=(lambda ch=ch: self.get_mem_mode(ch)),
                set_cmd=(lambda v, ch=ch: self.set_mem_mode(ch, v)),
                vals=Enum("int", "ext", "INT", "EXT"),
            )

    def close(self) -> None:
        """`Instrument.close()` only unregisters this instrument from
        qcodes' instrument registry - it has no VISA resource of its own
        to release. Close the actual connection too."""
        _RawM8195A.close(self)
        Instrument.close(self)
