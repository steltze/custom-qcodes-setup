"""Framework-agnostic instrument driver plumbing.

Talks raw VISA/SCPI over pyvisa only - nothing in this package imports
qcodes, exopy, or pyarbtools. Drivers built on `VisaDriver` expose their
settings as plain `get_x()`/`set_x(value)` methods (not descriptor-based
properties): a `frequency` data descriptor would collide with qcodes'
`add_parameter`, which needs to bind its own `Parameter` object to that
exact attribute name on a merged instance - see `instruments_native/`'s
module docstrings for the full story, including a worse reason this was
tried and reverted: a `self.frequency = value` written *inside the
driver's own methods* (e.g. `configure()`/`safe_shutdown()`) would have
silently stopped reaching the instrument at all once shadowed, with no
error - plain methods can't have that failure mode.

For qcodes Station/Measurement use, `instruments_native/*.py` builds one
instrument class per driver via multiple inheritance directly on top of
the driver class (`class Foo(drivers.foo.Foo, qcodes.instrument.
Instrument)`) rather than wrapping a driver *instance* inside a separate
qcodes instrument object - see that package for the base-order and
name-collision details that requires.
"""

from __future__ import annotations

import time
from typing import Any


class VisaDriver:
    """Common connection, `*IDN?`, error-queue, and connect-banner
    plumbing shared by every driver in this package.

    Subclasses open no connection of their own - `__init__` here does it -
    and are expected to call `self.connect_message()` once at the end of
    their own `__init__`, after any startup writes/config have gone out,
    matching the point qcodes' own drivers call it.
    """

    default_terminator = "\n"
    default_timeout = 10_000  # ms

    def __init__(
        self,
        name: str,
        address: str,
        *,
        terminator: str | None = None,
        timeout: float | None = None,
        visalib: str | None = None,
    ) -> None:
        import pyvisa  # deferred: no import cost for callers that never connect

        if not hasattr(self, "_t0"):
            # Already set by qcodes' InstrumentBase.__init__ when this
            # class is merged into a qcodes-native instrument (see
            # instruments_native/) - that one ran first and is a better
            # start time for connect_message() anyway (it predates this
            # __init__, i.e. the VISA connection itself).
            self._t0 = time.time()
        if not hasattr(self, "name"):
            # qcodes' InstrumentBase defines `name` as a read-only
            # property with no setter - when merged with a qcodes
            # instrument (Instrument.__init__ having already run first),
            # `self.name` already works and must not be reassigned here.
            self.name = name
        self.address = address

        self._resource_manager = (
            pyvisa.ResourceManager(visalib) if visalib else pyvisa.ResourceManager()
        )
        self.resource = self._resource_manager.open_resource(address)

        term = self.default_terminator if terminator is None else terminator
        self.resource.write_termination = term
        self.resource.read_termination = term
        self.resource.timeout = self.default_timeout if timeout is None else timeout

    # -- raw I/O ------------------------------------------------------------
    def write(self, cmd: str) -> None:
        self.resource.write(cmd)

    def ask(self, cmd: str) -> str:
        return self.resource.query(cmd).strip()

    def close(self) -> None:
        self.resource.close()

    # -- identification -------------------------------------------------------
    def get_idn(self) -> dict[str, str | None]:
        """Parse `*IDN?` into vendor/model/serial/firmware. Same splitting
        rules as `qcodes.instrument.Instrument.get_idn` (comma normally,
        semicolon/colon accepted too), so the banner below reads exactly
        like a qcodes connect message would."""
        idstr = ""
        try:
            idstr = self.ask("*IDN?")
            idparts: list[str | None] = []
            for separator in ",;:":
                idparts = [p.strip() for p in idstr.split(separator, 3)]
                if len(idparts) > 1:
                    break
            if len(idparts) < 4:
                idparts += [None] * (4 - len(idparts))
        except Exception:
            idparts = [None, self.name, None, None]

        if str(idparts[1]).lower().startswith("model"):
            idparts[1] = str(idparts[1])[5:].strip()

        return dict(zip(("vendor", "model", "serial", "firmware"), idparts))

    def connect_message(self, begin_time: float | None = None) -> None:
        """Print the same "Connected to: ..." banner
        `qcodes.instrument.Instrument.connect_message` does, so connecting
        through a bare driver feels identical to connecting through the
        qcodes wrapper."""
        idn: dict[str, str | None] = {
            "vendor": None,
            "model": None,
            "serial": None,
            "firmware": None,
        }
        idn.update(self.get_idn())
        t = time.time() - (begin_time or self._t0)
        print(
            "Connected to: {vendor} {model} "
            "(serial:{serial}, firmware:{firmware}) "
            "in {t:.2f}s".format(t=t, **idn)
        )

    # -- error queue ----------------------------------------------------------
    def get_error(self, cmd: str = ":SYST:ERR?") -> str:
        """Pop one entry off the instrument's error queue."""
        return self.ask(cmd)

    def flush_errors(self, cmd: str = ":SYST:ERR?", limit: int = 50) -> list[str]:
        """Drain the error queue; returns everything that was queued."""
        errors = []
        for _ in range(limit):
            err = self.get_error(cmd)
            if err.startswith(("0,", "+0,")):
                break
            errors.append(err)
        return errors

    # -- measurement-harness compatibility -------------------------------------
    def get_config_info(self) -> dict[str, Any]:
        """Same shape `instruments_old.basic_instrument.BasicInstrument`
        and every qcodes wrapper in `instruments/` already return
        (`visa_address`/`nickname`/`config`), so this driver can be handed
        straight to `BaseMeasurement.instruments` / `qcodes_utils.
        measurement_run` helpers, with or without a qcodes wrapper on top.
        """
        return {
            "visa_address": self.address,
            "nickname": self.name,
            "config": getattr(self, "_config", {}),
        }
