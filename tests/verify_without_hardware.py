"""
Exercise KeysightM8195A and AnaPicoAPUASYN20 - including the real
pyarbtools/exopy drivers they wrap - and a full QCoDeS Measurement/
datasaver run, all without touching real hardware.

How: both legacy drivers open their connection with
`pyvisa.ResourceManager().open_resource(...)`. This monkeypatches that one
call to return a small hand-rolled fake VISA resource per instrument that
answers only the exact SCPI dialect pyarbtools/exopy send (read directly
out of their installed source - see the class docstrings below for what
that dialect is). Everything above the socket - pyarbtools' configure()/
set_*(), exopy's AnapicoChannel properties, and our QCoDeS Parameters -
then runs for real.

This is a hand-rolled, lighter-weight stand-in for `pyvisa-sim` (a
YAML-driven VISA simulator, the standard tool for this). It is *not*
verified against the real instruments' firmware - only against the
client-library source - so a green run here proves "the Python plumbing
is correct", not "the SCPI is correct". Swap in pyvisa-sim (or real
hardware) for the latter.

Run with: .venv/bin/python tests/verify_without_hardware.py
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from instruments.AnaPicoAPUASYN20 import AnaPicoAPUASYN20  # noqa: E402
from instruments.KeysightM8195A import KeysightM8195A  # noqa: E402


class FakeM8195AResource:
    """Stand-in for the M8195A's SCPI dialect - only the commands
    `pyarbtools.instruments.M8195A` actually sends (`inst:dacm`,
    `frequency:raster`, `func:mode`, `roscillator:source/frequency`,
    `voltage1..4`, `instrument:memory:extended:rdivider`, plus
    `*rst`/`*opc?`/`*idn?`/`SYST:ERR?`/waveform trace commands).
    State-tracking: a write updates internal state, a query reflects it.
    """

    def __init__(self) -> None:
        self.timeout = 0
        self._state = {
            "inst:dacm": "DUAL",
            "frequency:raster": "65000000000.0",
            "func:mode": "ARB",
            "roscillator:source": "EXT",
            "roscillator:frequency": "10000000.0",
            "voltage1": "1.0",
            "voltage2": "1.0",
            "voltage3": "1.0",
            "voltage4": "1.0",
        }
        self._mem_div = 1
        self._segment_counter = 0

    def write(self, cmd: str) -> None:
        low = cmd.strip().lower()
        if low in ("*rst", "abort"):
            return
        if low.startswith(("output", "trace:select", "init:cont", "init:imm")):
            return
        if low.startswith("trace") and (":def" in low or ":name" in low):
            return
        if low.startswith("instrument:memory:extended:rdivider"):
            self._mem_div = int(low.rsplit("div", 1)[-1])
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
            return "FAKE,M8195A,0,0.0"
        if low == "*opc":
            return "1"
        if low == "syst:err":
            return '0,"No error"'
        if low == "instrument:memory:extended:rdivider":
            return f"DIV{self._mem_div}"
        if "catalog" in low:
            self._segment_counter += 1
            return f"seg,{self._segment_counter - 1},end"
        if low in self._state:
            return self._state[low]
        raise ValueError(f"FakeM8195AResource: unhandled query {cmd!r}")

    def close(self) -> None:
        pass


class FakeAnapicoResource:
    """Stand-in for the Anapico's SCPI dialect, transcribed from
    instruments_old/exopy_hqc_legacy/drivers/visa/anapico.py: per-channel
    `:SOURn:FREQ/POWER/PHAS`, `:OUTPn?` (get) / `:OUTPUTn ON|OFF` (set),
    and the shared `:SOUR:ROSC:SOUR` / `:SOUR:ROSC:LOCK?`.
    """

    def __init__(self) -> None:
        self.timeout = 0
        self.query_delay = 0
        self.write_termination = "\n"
        self.read_termination = "\n"
        self._freq = {1: 1e9}
        self._power = {1: -10.0}
        self._phase = {1: 0.0}
        self._output = {1: False}
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
            return "FAKE,Anapico,0,0.0"
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


class _FakeResourceManager:
    def __init__(self, factory) -> None:
        self._factory = factory

    def open_resource(self, address: str, **kwargs):
        return self._factory()


def main() -> None:
    with (
        patch("pyvisa.ResourceManager", lambda *a, **kw: _FakeResourceManager(FakeM8195AResource)),
        patch(
            "instruments_old.exopy_hqc_legacy.drivers.visa_tools.ResourceManager",
            lambda *a, **kw: _FakeResourceManager(FakeAnapicoResource),
        ),
    ):
        awg = KeysightM8195A("awg_sim", "169.254.1.1")
        anapico = AnaPicoAPUASYN20("anapico_sim", "TCPIP::169.254.1.2::INSTR")

        # -- parameter round-trips, through the real wrapped drivers -------
        assert awg.dac_mode() == "dual"
        awg.sample_rate(60e9)
        assert awg.sample_rate() == 60e9
        awg.amplitude_1(0.5)
        assert awg.amplitude_1() == 0.5

        anapico.frequency(2.4e9)
        assert anapico.frequency() == 2.4e9
        anapico.output_enabled(True)
        assert anapico.output_enabled() is True
        assert anapico.reference_source() == "EXT"

        # -- waveform path (pyarbtools' granularity/padding logic) --------
        seg_id = awg.send_sine(freq=1e9, phase=0.0, channel=1, amp=0.8)
        assert isinstance(seg_id, int)
        assert awg.amplitude_1() == 0.8

        print("Parameter and waveform round-trips: PASS")

        # -- save data the way QCoDeS intends -------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            from qcodes.dataset import (
                Measurement,
                initialise_or_create_database_at,
                load_or_create_experiment,
            )

            db_path = Path(tmpdir) / "verify.db"
            initialise_or_create_database_at(db_path)
            load_or_create_experiment(
                experiment_name="verify_without_hardware", sample_name="fake"
            )

            meas = Measurement()
            meas.register_parameter(anapico.frequency)
            meas.register_parameter(awg.sample_rate, setpoints=(anapico.frequency,))

            with meas.run() as datasaver:
                for freq in (1e9, 2e9, 3e9):
                    anapico.frequency(freq)
                    datasaver.add_result(
                        (anapico.frequency, anapico.frequency()),
                        (awg.sample_rate, awg.sample_rate()),
                    )
                dataset = datasaver.dataset

            df = dataset.to_pandas_dataframe()
            assert len(df) == 3
            print(f"Measurement/datasaver round-trip: PASS ({len(df)} rows written)")
            print(df)


if __name__ == "__main__":

    from qcodes.logger import start_all_logging

    start_all_logging()


    main()
