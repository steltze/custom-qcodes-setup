"""AnaPico APUASYN20 - qcodes-compatible instrument.

One instrument class, built via multiple inheritance directly on top of
`native.drivers.anapico_apuasyn20.AnaPicoAPUASYN20` - not a `qcodes.Instrument`
holding a separate driver object as a delegate (`self._driver = ...`).
`isinstance(anapico, native.drivers.anapico_apuasyn20.AnaPicoAPUASYN20)` is True,
so anything that only needs the plain driver surface
(`get_config_info`/`safe_shutdown`/`close`/`get_x()`/`set_x()`) works
directly; `isinstance(anapico, qcodes.instrument.Instrument)` is also
True, so it plugs straight into a qcodes Station and
`Measurement.register_parameter` like any other qcodes instrument.

Base order matters: `_RawAnaPico` is listed *before* `Instrument` so that
plain `self.write`/`self.ask` resolve to the driver's real pyvisa-backed
versions. Plain `qcodes.instrument.Instrument` (unlike `VisaInstrument`)
defines its own `write`/`ask` that unconditionally raise
`NotImplementedError` via `write_raw`/`ask_raw` - if `Instrument` came
first in the MRO, every SCPI call the driver makes internally (including
`get_idn`/`connect_message`, both plain methods that end up calling
`self.ask`) would hit that stub instead. With the driver first, its own
`get_idn`/`connect_message`/`get_config_info`/`safe_shutdown` are what
run too - correct either way, just self-contained rather than routed
through qcodes' `IDN` Parameter. `close()` still needs an explicit
override below: whichever single `close()` the MRO would pick only does
half the job (either the VISA teardown or the qcodes instrument-registry
teardown, not both).

(Why the driver exposes `get_frequency()`/`set_frequency()` methods
rather than a `frequency` property: see `native/drivers/base.py`'s module
docstring. In short, a class-level data descriptor of that name anywhere
in the MRO would intercept `add_parameter`'s attempt to bind
`self.frequency` to a qcodes `Parameter` on this merged instance - and,
worse, would keep intercepting `self.frequency = ...` inside the driver's
*own* methods like `configure()`/`safe_shutdown()` too, silently turning
those into no-ops instead of real SCPI writes. Plain methods don't have
that failure mode.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.validators import Bool, Enum, Numbers

from native.drivers.anapico_apuasyn20 import AnaPicoAPUASYN20 as _RawAnaPico


class AnaPicoAPUASYN20(_RawAnaPico, Instrument):
    """Single-channel Anapico APUASYN20, 8 kHz - 20 GHz, as a qcodes
    instrument. See `native.drivers.anapico_apuasyn20.AnaPicoAPUASYN20` for the
    underlying SCPI and the real-hardware notes behind e.g. the power
    range."""

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        Instrument.__init__(self, name, **kwargs)
        _RawAnaPico.__init__(self, name, address, config=config)

        self.frequency: Parameter = self.add_parameter(
            "frequency",
            label="Frequency",
            unit="Hz",
            get_cmd=self.get_frequency,
            set_cmd=self.set_frequency,
            vals=Numbers(min_value=0),
        )

        self.power: Parameter = self.add_parameter(
            "power",
            label="Output power",
            unit="dBm",
            get_cmd=self.get_power,
            set_cmd=self.set_power,
            # `set_power` (native/drivers/anapico_apuasyn20.py) already enforces
            # this range itself - kept visible here too so it shows up on
            # the qcodes Parameter/snapshot.
            vals=Numbers(min_value=-10.0, max_value=23.0),
        )

        self.phase: Parameter = self.add_parameter(
            "phase",
            label="Phase",
            unit="rad",
            get_cmd=self.get_phase,
            set_cmd=self.set_phase,
            vals=Numbers(-np.pi, np.pi),
        )

        self.output_enabled: Parameter = self.add_parameter(
            "output_enabled",
            label="Output enabled",
            get_cmd=self.get_output_enabled,
            set_cmd=self.set_output_enabled,
            vals=Bool(),
        )

        self.reference_source: Parameter = self.add_parameter(
            "reference_source",
            label="Reference oscillator source",
            get_cmd=self.get_reference_source,
            set_cmd=self.set_reference_source,
            vals=Enum("INT", "EXT"),
        )

        self.oscillator_locked: Parameter = self.add_parameter(
            "oscillator_locked",
            label="Reference oscillator locked",
            get_cmd=self.get_oscillator_locked,
            vals=Bool(),
        )

    def close(self) -> None:
        """`Instrument.close()` only unregisters this instrument from
        qcodes' instrument registry - it has no VISA resource of its own
        to release. Close the actual connection too."""
        _RawAnaPico.close(self)
        Instrument.close(self)
