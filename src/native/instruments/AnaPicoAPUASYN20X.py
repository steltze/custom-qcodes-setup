"""AnaPico APUASYN20-X (4 channels) - qcodes-compatible instrument.

Sibling of `native/instruments/AnaPicoAPUASYN20.py` (single-channel) -
see that module's docstring for why this is built via multiple
inheritance directly on `native.drivers.anapico_apuasyn20x.AnaPicoAPUASYN20X`
(base order, the `name`/`write`/`ask` collisions with plain
`qcodes.instrument.Instrument`, and the explicit `close()` override),
rather than a wrapper holding a separate driver object.

Channel numbers are 1-indexed (`frequency_1`.. `frequency_4`), matching
the driver and every other multi-channel instrument in this repo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.validators import Bool, Enum, Numbers

from native.drivers.anapico_apuasyn20x import CHANNELS
from native.drivers.anapico_apuasyn20x import AnaPicoAPUASYN20X as _RawAnaPicoX


class AnaPicoAPUASYN20X(_RawAnaPicoX, Instrument):
    """4-channel Anapico APUASYN20-X, 8 kHz - 20 GHz per channel, as a
    qcodes instrument. See `native.drivers.anapico_apuasyn20x.AnaPicoAPUASYN20X`
    for the underlying SCPI."""

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        Instrument.__init__(self, name, **kwargs)
        _RawAnaPicoX.__init__(self, name, address, config=config)

        for ch in CHANNELS:
            self.add_parameter(
                f"frequency_{ch}",
                label=f"Ch{ch} frequency",
                unit="Hz",
                get_cmd=(lambda ch=ch: self.get_frequency(ch)),
                set_cmd=(lambda v, ch=ch: self.set_frequency(ch, v)),
                vals=Numbers(min_value=0),
            )
            self.add_parameter(
                f"power_{ch}",
                label=f"Ch{ch} output power",
                unit="dBm",
                get_cmd=(lambda ch=ch: self.get_power(ch)),
                set_cmd=(lambda v, ch=ch: self.set_power(ch, v)),
                # `set_power` (native/drivers/anapico_apuasyn20x.py) already
                # enforces this range itself - kept visible here too so
                # it shows up on the qcodes Parameter/snapshot.
                vals=Numbers(min_value=-10.0, max_value=23.0),
            )
            self.add_parameter(
                f"phase_{ch}",
                label=f"Ch{ch} phase",
                unit="rad",
                get_cmd=(lambda ch=ch: self.get_phase(ch)),
                set_cmd=(lambda v, ch=ch: self.set_phase(ch, v)),
                vals=Numbers(-np.pi, np.pi),
            )
            self.add_parameter(
                f"output_enabled_{ch}",
                label=f"Ch{ch} output enabled",
                get_cmd=(lambda ch=ch: self.get_output_enabled(ch)),
                set_cmd=(lambda v, ch=ch: self.set_output_enabled(ch, v)),
                vals=Bool(),
            )

        # Shared across all 4 channels, not per-channel.
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
        _RawAnaPicoX.close(self)
        Instrument.close(self)
