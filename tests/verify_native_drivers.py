"""
Exercise the new, detached-from-qcodes driver package (`native/drivers/`)
and its qcodes-compatible instruments (`native/instruments/`), both
standalone and merged, without touching real hardware. Companion to
`verify_without_hardware.py`, which covers the still-legacy-plugged
`legacy/instruments/` layer instead - that one is deliberately left alone.

How: `native.drivers.anapico_apuasyn20.AnaPicoAPUASYN20` opens its connection
with `pyvisa.ResourceManager().open_resource(address)`, exactly like every
other pyvisa-based driver in this repo. This monkeypatches that one call
to return a small hand-rolled fake VISA resource answering the exact SCPI
dialect transcribed into the driver (see its module docstring and
`legacy/drivers/exopy_hqc_legacy/drivers/visa/anapico.py`, which it was
transcribed from). Everything above the socket - the driver's own
get_x()/set_x() methods, and the qcodes Parameters wrapping them in
`native.instruments.AnaPicoAPUASYN20` - then runs for real.

What this specifically guards against: `native.instruments.
AnaPicoAPUASYN20` is built via multiple inheritance directly on the raw
driver class (`class AnaPicoAPUASYN20(_RawAnaPico, Instrument)`), not a
wrapper holding a separate driver instance - see that module's docstring
for why base order and a `close()` override are both required for this to
work at all. An earlier version of this pattern used descriptor-based
properties instead of get_x()/set_x() methods and had a *silent*
correctness bug: `safe_shutdown()`/`configure()` (defined on the driver,
called via `self`) stopped reaching the instrument entirely on a merged
instance - no error, the RF output just never actually turned off. The
`safe_shutdown()` and `config=` checks below exist specifically to catch
that failure mode if it's ever reintroduced.

This is a hand-rolled, lighter-weight stand-in for `pyvisa-sim` (a
YAML-driven VISA simulator, the standard tool for this) - not verified
against the real instrument's firmware, only against the SCPI dialect
transcribed from the exopy driver (some of which, in turn, came from
real-hardware findings - see `native/drivers/anapico_apuasyn20.py`). A green run
here proves "the Python plumbing is correct", not "the SCPI is correct".

Run with: .venv/bin/python tests/verify_native_drivers.py
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logger = logging.getLogger(__name__)


class FakeAnapicoResource:
    """Stand-in for the Anapico's SCPI dialect - see
    `native/drivers/anapico_apuasyn20.py`'s module docstring for where each
    command came from."""

    def __init__(self) -> None:
        self.timeout = 0
        self.query_delay = 0
        self.write_termination = "\n"
        self.read_termination = "\n"
        # Channels 1-4 all pre-populated so this one fake resource serves
        # both the single-channel driver (only ever touches channel 1)
        # and the -X 4-channel driver.
        self._freq = {ch: 1e9 for ch in (1, 2, 3, 4)}
        self._power = {ch: -10.0 for ch in (1, 2, 3, 4)}
        self._phase = {ch: 0.0 for ch in (1, 2, 3, 4)}
        self._output = {ch: False for ch in (1, 2, 3, 4)}
        self._ref_source = "INT"

    def write(self, cmd: str) -> None:
        cmd = cmd.strip()
        if m := re.match(r":SOUR(\d+):FREQ ([\-0-9.eE]+)", cmd, re.I):
            self._freq[int(m.group(1))] = float(m.group(2))
        elif m := re.match(r":SOUR(\d+):POWER ([\-0-9.eE]+)", cmd, re.I):
            self._power[int(m.group(1))] = float(m.group(2))
        elif m := re.match(r":SOUR(\d+):PHAS ([\-0-9.eE]+)", cmd, re.I):
            self._phase[int(m.group(1))] = float(m.group(2))
        elif m := re.match(r":OUTPUT(\d+) (ON|OFF)", cmd, re.I):
            self._output[int(m.group(1))] = m.group(2).upper() == "ON"
        elif m := re.match(r":SOUR:ROSC:SOUR (EXT|INT)", cmd, re.I):
            self._ref_source = m.group(1).upper()
        # anything else (e.g. SYST:COMM:VXI:RTMO 0) is an accepted no-op

    def query(self, cmd: str) -> str:
        cmd = cmd.strip()
        if re.match(r"\*IDN\?", cmd, re.I):
            return "FAKE,Anapico,SN123,0.4.106"
        if m := re.match(r":SOUR(\d+):FREQ\?", cmd, re.I):
            return str(self._freq[int(m.group(1))])
        if m := re.match(r":SOUR(\d+):POWER\?", cmd, re.I):
            return str(self._power[int(m.group(1))])
        if m := re.match(r":SOUR(\d+):PHAS\?", cmd, re.I):
            return str(self._phase[int(m.group(1))])
        if m := re.match(r":OUTP(?:UT)?(\d+)\?", cmd, re.I):
            return "1" if self._output[int(m.group(1))] else "0"
        if re.match(r":SOUR:ROSC:SOUR\?", cmd, re.I):
            return self._ref_source
        if re.match(r":SOUR:ROSC:LOCK\?", cmd, re.I):
            return "1"
        if re.match(r":SYST:ERR\?", cmd, re.I):
            return '0,"No error"'
        raise ValueError(f"FakeAnapicoResource: unhandled query {cmd!r}")

    def close(self) -> None:
        pass


class FakeM8195AResource:
    """Stand-in for the M8195A's SCPI dialect that
    `native.drivers.keysight_m8195a.KeysightM8195A` sends - transcribed from the
    installed `pyarbtools.instruments.M8195A`, see that driver's module
    docstring. State-tracking: a write updates internal state, a query
    reflects it. `trace<ch>:catalog?` is simulated with a single global
    incrementing counter rather than a real per-channel catalog - enough
    to exercise `download_wfm`'s segment-numbering logic, not a real
    catalog simulation."""

    def __init__(self) -> None:
        self.timeout = 0
        self._state = {
            "inst:dacm": "DUAL",
            "frequency:raster": "65000000000.0",
            "func:mode": "ARB",
            "roscillator:source": "EXT",
            "roscillator:frequency": "10000000.0",
            "outp:rosc:source": "INT",
            "voltage1": "1.0",
            "voltage2": "1.0",
            "voltage3": "1.0",
            "voltage4": "1.0",
        }
        self._mem_div = 1
        self._output = {1: False, 2: False, 3: False, 4: False}
        self._mem_mode = {1: "EXT", 2: "EXT", 3: "EXT", 4: "EXT"}
        self._fir_scale = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        self._segment_counter = 0

    def write(self, cmd: str) -> None:
        low = cmd.strip().lower()
        if low in ("*rst", "abort"):
            return
        if m := re.match(r"output(\d) (on|off)", low):
            self._output[int(m.group(1))] = m.group(2) == "on"
            return
        if low.startswith(("trace:select", "init:cont", "init:imm")):
            return
        if low.startswith("trace") and (":def" in low or ":name" in low):
            return
        if low.startswith("instrument:memory:extended:rdivider"):
            self._mem_div = int(low.rsplit("div", 1)[-1])
            return
        if m := re.match(r"trac(\d):mmod (int|ext)", low):
            self._mem_mode[int(m.group(1))] = m.group(2).upper()
            return
        if m := re.match(r"outp(\d):filt:\w+:scal ([\-0-9.eE]+)", low):
            self._fir_scale[int(m.group(1))] = float(m.group(2))
            return
        for key in self._state:
            if low.startswith(key + " "):
                self._state[key] = cmd[len(key):].strip().upper()
                return
        raise ValueError(f"FakeM8195AResource: unhandled write {cmd!r}")

    def write_binary_values(self, cmd: str, data, datatype: str = "b") -> None:
        pass

    def query(self, cmd: str) -> str:
        low = cmd.strip().lower().rstrip("?")
        if low == "*idn":
            return "FAKE,M8195A,SN789,1.2.3"
        if low == "*opc":
            return "1"
        if low == "syst:err":
            return '0,"No error"'
        if low == "instrument:memory:extended:rdivider":
            return f"DIV{self._mem_div}"
        if m := re.match(r"output(\d)", low):
            return "1" if self._output[int(m.group(1))] else "0"
        if m := re.match(r"trac(\d):mmod", low):
            return self._mem_mode[int(m.group(1))]
        if m := re.match(r"outp(\d):filt:\w+:scal", low):
            return str(self._fir_scale[int(m.group(1))])
        if "catalog" in low:
            self._segment_counter += 1
            return f"seg,{self._segment_counter - 1},end"
        if low in self._state:
            return self._state[low]
        raise ValueError(f"FakeM8195AResource: unhandled query {cmd!r}")

    def close(self) -> None:
        pass


class _FakeResourceManager:
    def __init__(self, factory=FakeAnapicoResource) -> None:
        self._factory = factory

    def open_resource(self, address: str, **kwargs):
        return self._factory()


def check_standalone_driver() -> None:
    from native.drivers.anapico_apuasyn20 import AnaPicoAPUASYN20 as RawAnaPico

    driver = RawAnaPico("anapico_raw", "TCPIP::169.254.1.2::INSTR")
    driver.set_frequency(2.4e9)
    assert driver.get_frequency() == 2.4e9
    driver.set_power(5.0)
    assert driver.get_power() == 5.0
    try:
        driver.set_power(50.0)
        raise AssertionError("expected ValueError for out-of-range power")
    except ValueError:
        pass
    driver.set_output_enabled(True)
    assert driver.get_output_enabled() is True
    assert driver.get_reference_source() == "INT"
    assert driver.get_oscillator_locked() is True
    driver.safe_shutdown()
    assert driver.get_output_enabled() is False
    driver.close()

    configured = RawAnaPico(
        "anapico_raw_configured",
        "TCPIP::169.254.1.4::INSTR",
        config={"frequency": 5e9, "power": 1.0, "output": True},
    )
    assert configured.get_frequency() == 5e9
    assert configured.get_power() == 1.0
    assert configured.get_output_enabled() is True
    configured.close()

    logger.info("standalone driver (native.drivers.anapico_apuasyn20): PASS")


def check_qcodes_native_instrument() -> None:
    import qcodes.instrument

    from native.drivers.anapico_apuasyn20 import AnaPicoAPUASYN20 as RawAnaPico
    from native.instruments.AnaPicoAPUASYN20 import AnaPicoAPUASYN20 as NativeAnaPico

    anapico = NativeAnaPico("anapico_native", "TCPIP::169.254.1.3::INSTR")
    assert isinstance(anapico, RawAnaPico), "must be a real instance of the raw driver"
    assert isinstance(anapico, qcodes.instrument.Instrument), "must be a real qcodes Instrument"

    anapico.frequency(3.3e9)
    assert anapico.frequency() == 3.3e9
    anapico.power(2.0)
    assert anapico.power() == 2.0
    anapico.output_enabled(True)
    assert anapico.output_enabled() is True
    assert anapico.reference_source() == "INT"
    assert anapico.oscillator_locked() is True
    assert anapico.IDN()["model"] == "Anapico"
    assert anapico.get_config_info()["nickname"] == "anapico_native"

    try:
        anapico.power(999)
        raise AssertionError("expected the qcodes Numbers validator to reject this")
    except Exception as exc:
        assert not isinstance(exc, AssertionError)

    # The failure mode an earlier (descriptor-based) version of this
    # pattern had: safe_shutdown(), defined on the driver and called via
    # `self`, silently stopped reaching the instrument on a merged
    # instance. Confirm it actually flips the real SCPI state.
    anapico.safe_shutdown()
    assert anapico.output_enabled() is False
    anapico.close()

    configured = NativeAnaPico(
        "anapico_native_configured",
        "TCPIP::169.254.1.5::INSTR",
        config={"frequency": 6e9, "power": -3.0, "output": True},
    )
    assert configured.frequency() == 6e9
    assert configured.power() == -3.0
    assert configured.output_enabled() is True
    configured.close()

    logger.info("qcodes-native instrument (native.instruments.AnaPicoAPUASYN20): PASS")


def check_qcodes_measurement_roundtrip() -> None:
    from qcodes.dataset import (
        Measurement,
        initialise_or_create_database_at,
        load_or_create_experiment,
    )

    from native.instruments.AnaPicoAPUASYN20 import AnaPicoAPUASYN20 as NativeAnaPico

    anapico = NativeAnaPico("anapico_meas", "TCPIP::169.254.1.6::INSTR")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "verify.db"
            initialise_or_create_database_at(db_path)
            load_or_create_experiment(
                experiment_name="verify_native_drivers", sample_name="fake"
            )

            meas = Measurement()
            meas.register_parameter(anapico.frequency)
            with meas.run() as datasaver:
                for freq in (1e9, 2e9, 3e9):
                    anapico.frequency(freq)
                    datasaver.add_result((anapico.frequency, anapico.frequency()))
                dataset = datasaver.dataset

            df = dataset.to_pandas_dataframe()
            assert len(df) == 3
            logger.info("Measurement/datasaver round-trip: PASS (%d rows written)", len(df))
    finally:
        anapico.close()


def check_standalone_driver_x() -> None:
    from native.drivers.anapico_apuasyn20x import AnaPicoAPUASYN20X as RawAnaPicoX

    driver = RawAnaPicoX(
        "picox_raw",
        "TCPIP::169.254.2.1::INSTR",
        config={
            "frequencies": (1e9, 2e9, 3e9, 4e9),
            "powers": (-5.0, -5.0, -5.0, -5.0),
            "outputs": ("OFF", "ON", "OFF", "OFF"),
        },
    )
    assert driver.get_frequency(1) == 1e9
    assert driver.get_frequency(4) == 4e9
    assert driver.get_power(2) == -5.0
    assert driver.which_outputs_enabled() == {1: False, 2: True, 3: False, 4: False}
    driver.safe_shutdown()
    assert driver.which_outputs_enabled() == {1: False, 2: False, 3: False, 4: False}
    driver.close()
    logger.info("standalone driver (native.drivers.anapico_apuasyn20x): PASS")


def check_qcodes_native_instrument_x() -> None:
    import qcodes.instrument

    from native.drivers.anapico_apuasyn20x import AnaPicoAPUASYN20X as RawAnaPicoX
    from native.instruments.AnaPicoAPUASYN20X import AnaPicoAPUASYN20X as NativeAnaPicoX

    anapico = NativeAnaPicoX(
        "picox_native",
        "TCPIP::169.254.2.2::INSTR",
        config={
            "frequencies": (1e9, 2e9, 3e9, 4e9),
            "powers": (-5.0, -5.0, -5.0, -5.0),
            "outputs": ("ON", "ON", "OFF", "OFF"),
        },
    )
    assert isinstance(anapico, RawAnaPicoX)
    assert isinstance(anapico, qcodes.instrument.Instrument)
    assert anapico.frequency_3() == 3e9
    anapico.frequency_3(9e9)
    assert anapico.frequency_3() == 9e9
    assert anapico.output_enabled_1() is True
    assert anapico.output_enabled_3() is False
    anapico.reference_source("EXT")
    assert anapico.reference_source() == "EXT"

    # Same regression this file's single-channel checks guard against:
    # safe_shutdown() must actually reach every channel.
    anapico.safe_shutdown()
    assert anapico.output_enabled_1() is False
    assert anapico.output_enabled_2() is False
    anapico.close()
    logger.info("qcodes-native instrument (native.instruments.AnaPicoAPUASYN20X): PASS")


def check_standalone_awg_driver() -> None:
    from native.drivers.keysight_m8195a import KeysightM8195A as RawAWG

    awg = RawAWG("awg_raw", "TCPIP0::169.254.3.1::inst0::INSTR")
    assert awg.get_dac_mode() == "dual"
    awg.set_sample_rate(60e9)
    assert awg.get_sample_rate() == 60e9
    awg.set_amplitude(1, 0.5)
    assert awg.get_amplitude(1) == 0.5
    assert awg.get_output_enabled(1) is False
    awg.set_output_enabled(1, True)
    assert awg.get_output_enabled(1) is True
    awg.set_output_enabled(1, False)

    segment = awg.send_sine(freq=1e9, phase=0.0, channel=1, amp=0.8)
    assert isinstance(segment, int)
    assert awg.get_amplitude(1) == 0.8
    assert awg.get_output_enabled(1) is True, "play() should turn the channel on"

    # FIR filter scale / memory mode - not in stock pyarbtools, see
    # native/drivers/keysight_m8195a.py's module docstring.
    assert awg.get_fir_scale(1) == 1.0
    awg.set_fir_scale(1, 50e-3 / 75e-3)
    assert abs(awg.get_fir_scale(1) - 50e-3 / 75e-3) < 1e-9
    try:
        awg.set_fir_scale(1, 1.5)
        raise AssertionError("expected ValueError for out-of-range fir_scale")
    except ValueError:
        pass
    awg.set_mem_mode(1, "int")
    assert awg.get_mem_mode(1) == "INT"
    awg.set_output_reference_source("ext")
    assert awg.get_output_reference_source() == "EXT"

    awg.configure({"sample_rate": 65e9})
    assert awg.get_output_enabled(1) is False, "configure() stops every channel first"

    awg.safe_shutdown()
    assert awg.get_output_enabled(1) is False
    assert awg.get_output_enabled(4) is False
    awg.close()
    logger.info("standalone driver (native.drivers.keysight_m8195a): PASS")


def check_qcodes_native_awg_instrument() -> None:
    import qcodes.instrument

    from native.drivers.keysight_m8195a import KeysightM8195A as RawAWG
    from native.instruments.KeysightM8195A import KeysightM8195A as NativeAWG

    awg = NativeAWG("awg_native", "TCPIP0::169.254.3.2::inst0::INSTR")
    assert isinstance(awg, RawAWG)
    assert isinstance(awg, qcodes.instrument.Instrument)

    awg.sample_rate(60e9)
    assert awg.sample_rate() == 60e9
    awg.amplitude_1(0.5)
    assert awg.amplitude_1() == 0.5

    segment = awg.send_sine(freq=1e9, phase=0.0, channel=1, amp=0.8)
    assert isinstance(segment, int)
    assert awg.amplitude_1() == 0.8
    assert awg.output_enabled_1() is True

    awg.fir_scale_1(50e-3 / 75e-3)
    assert abs(awg.fir_scale_1() - 50e-3 / 75e-3) < 1e-9
    awg.mem_mode_1("int")
    assert awg.mem_mode_1() == "INT"
    awg.output_reference_source("ext")
    assert awg.output_reference_source() == "EXT"

    # Same regression the Anapico checks above guard against.
    awg.safe_shutdown()
    assert awg.output_enabled_1() is False
    assert awg.output_enabled_4() is False
    awg.close()
    logger.info("qcodes-native instrument (native.instruments.KeysightM8195A): PASS")


def main() -> None:
    with patch(
        "pyvisa.ResourceManager",
        lambda *a, **kw: _FakeResourceManager(FakeAnapicoResource),
    ):
        check_standalone_driver()
        check_qcodes_native_instrument()
        check_qcodes_measurement_roundtrip()
        check_standalone_driver_x()
        check_qcodes_native_instrument_x()

    with patch(
        "pyvisa.ResourceManager",
        lambda *a, **kw: _FakeResourceManager(FakeM8195AResource),
    ):
        check_standalone_awg_driver()
        check_qcodes_native_awg_instrument()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
