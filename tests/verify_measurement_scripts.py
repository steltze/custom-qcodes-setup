"""Exercise the renewed `measurement_scripts/*.py` end to end - real QCoDeS
`Measurement`/datasaver run into a sqlite `.db`, plus the h5-export step -
without touching real hardware.

How: `KeysightP5024A` extends QCoDeS' own `VisaInstrument`, which insists
the object `pyvisa.ResourceManager().open_resource(...)` returns is a real
`pyvisa.resources.MessageBasedResource` (checked with `isinstance`) - a
plain hand-rolled stand-in fails that check before a single command is even
sent (unlike the AWG/Anapico stopgap wrappers in
`tests/verify_without_hardware.py`, which never go through QCoDeS'
`VisaInstrument` at all). So instead of patching `pyvisa.ResourceManager`,
this monkeypatches `VisaInstrument._open_resource` directly and hands back
a fake resource object - qcodes never actually checks its type once that
method is bypassed.

`FakePNAResource` answers the exact SCPI subset `KeysightP5024A` (built on
qcodes' native `KeysightPNAxBase`/N52xx driver) sends - reverse-engineered
by tracing a real construction against a logging stand-in, not guessed. In
particular it emulates the trace-catalog/`add_trace()`/`CALC:PAR:MNUM`
bookkeeping the native driver relies on, and completes any group/single
sweep instantly (no real sweep timing) so `run_sweep()`'s poll-until-HOLD
loop returns immediately. Like `verify_without_hardware.py`, a green run
here proves "the Python plumbing (parameters, sweep loop, db write, h5
export) is correct", not "the SCPI is correct against real firmware".

Run with: .venv/bin/python tests/verify_measurement_scripts.py
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import struct
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py as h5
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class FakePNAResource:
    """Stand-in for the P5024A's SCPI dialect, as sent by qcodes'
    `KeysightPNAxBase`/N52xx driver plus this repo's `KeysightP5024A`
    extensions (`power_slope`, `configure_measurements`). Starts with one
    factory-default trace, like a real PNA does after power-on/preset."""

    def __init__(self) -> None:
        self.timeout = 0
        self.query_delay = 0
        self.write_termination = "\n"
        self.read_termination = "\n"
        # Minimal stand-ins so VisaInstrument's own bookkeeping (backend
        # detection, device_clear, close) doesn't need a real pyvisa
        # resource underneath.
        self.visalib = SimpleNamespace()
        self.session = 1
        self.resource_name = "TCPIP::169.254.1.3::INSTR"

        self._next_num = 1
        self._traces: list[dict] = []  # [{"num", "name", "sparam"}], catalog order
        self._active_num: int | None = None
        self._make_trace("S11")  # factory default, like real power-on state

        self._output = True
        self._power = -40.0
        self._power_slope = 0.0
        self._power_slope_state = 0
        self._if_bandwidth = 5000.0
        self._points = 2001
        self._start = 1e9
        self._stop = 20e9
        self._cw = 1e9
        self._sweep_type = "LIN"
        self._sweep_mode = "HOLD"
        self._averages_enabled = False
        self._averages = 1
        self._group_trigger_count = 1
        # KeysightPNABase.__init__ itself sends "FORM REAL,32" at connect
        # time (see N52xx.py) - match that as the starting state.
        self._data_format = "REAL,32"

    # -- trace bookkeeping ---------------------------------------------
    def _make_trace(self, sparam: str) -> dict:
        return self._make_trace_named(f"CH1_{sparam}_{self._next_num}", sparam)

    def _make_trace_named(self, name: str, sparam: str) -> dict:
        trace = {"num": self._next_num, "name": name, "sparam": sparam}
        self._next_num += 1
        self._traces.insert(0, trace)  # see add_trace()'s zip-diff note below
        self._active_num = trace["num"]
        return trace

    def _trace_by_num(self, num: int) -> dict:
        return next(t for t in self._traces if t["num"] == int(num))

    def _trace_by_name(self, name: str) -> dict:
        return next(t for t in self._traces if t["name"] == name)

    def _active_trace(self) -> dict:
        return self._trace_by_num(self._active_num)

    def _catalog(self) -> str:
        parts = []
        for t in self._traces:
            parts += [t["name"], t["sparam"]]
        return '"' + ",".join(parts) + '"'

    # -- pyvisa-shaped interface -----------------------------------------
    def write(self, cmd: str) -> None:
        cmd = cmd.strip().lstrip(":")
        low = cmd.lower()

        if low in ("*rst", "*cls"):
            return
        if low == "syst:preset":
            return
        if low.startswith("form:bord"):
            return  # byte-order sub-command - not tracked separately
        if low.startswith("form "):
            # e.g. "FORM REAL,32" / "FORM REAL,64" / "FORM ASC,0" - tracked
            # for real, not ignored: query_binary_values() below actually
            # encodes according to this, so a script that sets one FORM but
            # reads back assuming another gets genuinely corrupted data,
            # same as it would on real hardware (this is exactly the class
            # of bug that shipped once already - see KeysightP5024A
            # .reset_config's NOTE on REAL,32 vs REAL,64).
            self._data_format = cmd.split(None, 1)[1].strip().upper()
            return
        if low.startswith("outp"):
            self._output = cmd.split()[-1].strip().upper() in ("1", "ON")
            return
        if low.startswith("sour:pow1:slop:stat"):
            self._power_slope_state = int(cmd.split()[-1])
            return
        if low.startswith("sour:pow1:slop"):
            self._power_slope = float(cmd.split()[-1])
            return
        if low.startswith("sour:pow"):
            self._power = float(cmd.split()[-1])
            return
        if low.startswith("sens:band"):
            self._if_bandwidth = float(cmd.split()[-1])
            return
        if low.startswith("sens:swe:poin"):
            self._points = int(cmd.split()[-1])
            return
        if low.startswith("sens:freq:star"):
            self._start = float(cmd.split()[-1])
            return
        if low.startswith("sens:freq:stop"):
            self._stop = float(cmd.split()[-1])
            return
        if low.startswith("sens:freq:cw"):
            self._cw = float(cmd.split()[-1])
            return
        if low.startswith("sens:swe:type"):
            self._sweep_type = cmd.split()[-1]
            return
        if low.startswith("sens:swe:gro:coun"):
            self._group_trigger_count = int(cmd.split()[-1])
            return
        if low.startswith("sens:aver:cle"):
            return
        if low.startswith("sens:aver"):
            self._averages_enabled = cmd.split()[-1] == "1"
            return
        if low.startswith("sens:swe:mode"):
            mode = cmd.split()[-1]
            # No real sweep timing: any triggered sweep completes instantly.
            self._sweep_mode = "HOLD" if mode in ("GRO", "SING") else mode
            return
        if low.startswith("calc:par:mnum"):
            self._active_num = int(cmd.split()[-1])
            return
        if low.startswith("calc:par:sel"):
            name = cmd.split(None, 1)[1].strip().strip("'")
            self._active_num = self._trace_by_name(name)["num"]
            return
        if low.startswith("calc:par:mod:ext"):
            sparam = cmd.split(None, 1)[1].strip().strip('"')
            self._active_trace()["sparam"] = sparam
            return
        if m := re.match(
            r"calc(?:ulate)?\d*:par(?:ameter)?:def(?:ine)?:ext(?:ended)?\s+'([^']*)'\s*,\s*'([^']*)'",
            cmd,
            re.I,
        ):
            # KeysightP5024A._create_trace() - confirmed on real hardware
            # this is what actually creates a trace on this instrument
            # (add_trace()'s "DISP:TRAC:NEW 0" below does not).
            self._make_trace_named(m.group(1), m.group(2))
            return
        if low.startswith("calc:par:del:all"):
            self._traces = []
            self._active_num = None
            return
        if m := re.match(r"calc(?:ulate)?:par(?:ameter)?:del(?:ete)?\s+'([^']+)'", cmd, re.I):
            self._traces = [t for t in self._traces if t["name"] != m.group(1)]
            return
        if low == "disp:trac:new 0":
            self._make_trace("S11")
            return
        if low.startswith("calc:form"):
            return  # trace display format (e.g. POLAR) - only matters for CALC:DATA?
        if low.startswith("disp"):
            return  # window/trace-feed/scale-coupling cosmetics
        raise ValueError(f"FakePNAResource: unhandled write {cmd!r}")

    def query(self, cmd: str) -> str:
        cmd = cmd.strip().lstrip(":")
        low = cmd.lower().rstrip("?")

        if low == "*idn":
            return "Keysight Technologies,P5024A,FAKE,1.0"
        if low == "*opc":
            return "1"
        if low == "syst:err":
            return '+0,"No error"'
        if low in ("outp", "output"):
            return "1" if self._output else "0"
        if low == "sour:pow1:slop:stat":
            return str(self._power_slope_state)
        if low == "sour:pow1:slop":
            return str(self._power_slope)
        if low == "sour:pow":
            return str(self._power)
        if low == "sens:band":
            return str(self._if_bandwidth)
        if low == "sens:swe:poin":
            return str(self._points)
        if low == "sens:freq:star":
            return str(self._start)
        if low == "sens:freq:stop":
            return str(self._stop)
        if low == "sens:freq:cw":
            return str(self._cw)
        if low == "sens:swe:type":
            return self._sweep_type
        if low == "sens:swe:gro:coun":
            return str(self._group_trigger_count)
        if low == "sens:aver":
            return "1" if self._averages_enabled else "0"
        if low == "sens:aver:coun":
            return str(self._averages)
        if low == "sens:swe:mode":
            return self._sweep_mode
        if low == "calc:par:mnum":
            return str(self._active_num)
        if low == "calc:par:cat:ext":
            return self._catalog()
        raise ValueError(f"FakePNAResource: unhandled query {cmd!r}")

    def query_binary_values(self, cmd: str, datatype: str = "f", is_big_endian: bool = True, container=list):
        low = cmd.strip().lower()
        if not low.startswith("calc:data?"):
            raise ValueError(f"FakePNAResource: unhandled binary query {cmd!r}")
        # Deterministic-but-distinct fake complex trace, seeded by S-parameter
        # and length by the currently configured number of sweep points.
        sparam = self._active_trace()["sparam"]
        seed = sum(ord(c) for c in sparam)
        n = self._points
        values: list[float] = []
        for i in range(n):
            values.append(float(seed + i))       # real part
            values.append(float(-seed - i * 0.5))  # imaginary part

        # Actually round-trip through real byte packing according to
        # whatever FORM is currently set on the (fake) instrument, then
        # unpack with whatever `datatype` the caller asked for - if those
        # disagree (e.g. instrument sends REAL,64 doubles but the caller
        # reads them as 4-byte floats), this produces genuinely
        # byte-misaligned garbage, same as it would on real hardware,
        # instead of silently handing back the right numbers regardless
        # of what was actually requested.
        instrument_dtype = {"REAL,32": "f", "REAL,64": "d"}.get(self._data_format)
        if instrument_dtype is None:
            raise ValueError(
                f"FakePNAResource: binary query while FORM={self._data_format!r} "
                "(only REAL,32/REAL,64 are simulated)"
            )
        endian = ">" if is_big_endian else "<"
        packed = struct.pack(f"{endian}{len(values)}{instrument_dtype}", *values)
        unpacked = [v[0] for v in struct.iter_unpack(f"{endian}{datatype}", packed)]
        return container(unpacked) if container is not list else unpacked

    def close(self) -> None:
        pass

    def clear(self) -> None:
        pass


class FakeYokogawaResource:
    """Stand-in for the GS200's SCPI dialect, as sent by qcodes' native
    `YokogawaGS200` driver: `OUTPUT`/`:SOUR:FUNC`/`:SOUR:RANG`/`:SOUR:LEV`.
    Every `current(value)`/`voltage(value)` set re-queries `:SOUR:RANG?`
    for range checking (see `YokogawaGS200._set_output`), so that one has
    to actually work, not just be accepted and ignored."""

    def __init__(self, point_delay: float = 0.0) -> None:
        self.timeout = 0
        self.visalib = SimpleNamespace()
        self.session = 1
        self.resource_name = "TCPIP::169.254.1.4::INSTR"
        self.write_termination = "\n"
        self.read_termination = "\n"

        self._mode = "CURR"
        self._range = 1e-3
        self._level = 0.0
        self._output = False
        self.write_log: list[str] = []
        # Only used by the crash-durability test below, to slow a sweep
        # down enough to reliably SIGKILL it mid-run - every real sweep
        # point sets the current, so sleeping there paces the whole loop.
        self._point_delay = point_delay

    def write(self, cmd: str) -> None:
        cmd = cmd.strip().lstrip(":")
        self.write_log.append(cmd)
        low = cmd.lower()
        if low in ("*rst", "*cls"):
            return
        if low == "output 1":
            self._output = True
            return
        if low == "output 0":
            self._output = False
            return
        if low.startswith("sour:func"):
            self._mode = cmd.split()[-1].strip().upper()
            return
        if low.startswith("sour:rang"):
            self._range = float(cmd.split()[-1])
            return
        if low.startswith("sour:lev"):
            self._level = float(cmd.split()[-1])
            if self._point_delay:
                time.sleep(self._point_delay)
            return
        raise ValueError(f"FakeYokogawaResource: unhandled write {cmd!r}")

    def query(self, cmd: str) -> str:
        cmd = cmd.strip().lstrip(":")
        low = cmd.lower().rstrip("?")
        if low == "*idn":
            return "Yokogawa,GS200,FAKE,1.0"
        if low == "*opt":
            return ""  # no /MON option installed
        if low == "output":
            return "1" if self._output else "0"
        if low == "sour:func":
            return self._mode
        if low == "sour:rang":
            return str(self._range)
        if low == "sour:lev":
            return str(self._level)
        raise ValueError(f"FakeYokogawaResource: unhandled query {cmd!r}")

    def close(self) -> None:
        pass

    def clear(self) -> None:
        pass


@contextlib.contextmanager
def patched_visa(resources: dict[str, object]):
    """Route every `VisaInstrument._open_resource(address, ...)` call to
    the fake resource registered for that address - lets one patch cover
    constructing several different fake instruments in the same script."""
    from qcodes.instrument.visa import VisaInstrument

    with patch.object(
        VisaInstrument, "_open_resource", lambda self, addr, lib: resources[addr]
    ):
        yield


class FakeM8195AResourceExt:
    """`verify_without_hardware.py`'s `FakeM8195AResource` was built before
    `KeysightM8195A` grew `delete_segment`/`clear_all_wfm`/`stop` - it
    doesn't answer `trace<ch>:del <id>`/`trace<ch>:del:all`. Wrap it rather
    than duplicate its (already-proven) segment/voltage state tracking.

    Also tracks `outp<ch>:filt:<rate>:scal` (FIR filter scale - see
    `native/drivers/keysight_m8195a.py`), which the inner fake predates too, and
    logs every write so a test can assert on the exact sequence of
    amplitude/fir_scale commands sent per sweep point (same `write_log`
    pattern as `FakeYokogawaResource` above)."""

    def __init__(self) -> None:
        from verify_without_hardware import FakeM8195AResource

        self._inner = FakeM8195AResource()
        self._fir_scale = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        self.write_log: list[str] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def write(self, cmd: str) -> None:
        cmd = cmd.strip()
        self.write_log.append(cmd)
        low = cmd.lower()
        if m := re.match(r"outp(\d):filt:\w+:scal ([\-0-9.eE]+)", low):
            self._fir_scale[int(m.group(1))] = float(m.group(2))
            return
        if "del" in low:
            return  # trace<ch>:del <id> / trace<ch>:del:all
        self._inner.write(cmd)

    def query(self, cmd: str) -> str:
        low = cmd.strip().lower()
        if m := re.match(r"outp(\d):filt:\w+:scal\?", low):
            return str(self._fir_scale[int(m.group(1))])
        return self._inner.query(cmd)

    def write_binary_values(self, *a, **kw):
        return self._inner.write_binary_values(*a, **kw)


def _open_p5024a_with_fake(name: str, address: str):
    from stock_instruments.KeysightP5024A import KeysightP5024A

    with patched_visa({address: FakePNAResource()}):
        return KeysightP5024A(name, address)


def main() -> None:
    vna = _open_p5024a_with_fake("vna_sim", "TCPIP::169.254.1.3::INSTR")

    vna.configure_measurements(
        (
            ("Sig1Sig1", "S41"),
            ("Sig2Sig1", "S21"),
            ("Sig1Sig2", "S43"),
            ("Sig2Sig2", "S23"),
        )
    )
    vna.points(11)
    vna.run_averaging()
    data = {label: vna.read_raw_data(label) for label in ("Sig1Sig1", "Sig2Sig1")}
    assert data["Sig1Sig1"].shape == (11,)
    assert data["Sig1Sig1"].dtype == complex
    assert not (data["Sig1Sig1"] == data["Sig2Sig1"]).all()
    logger.info("KeysightP5024A parameter/trace round-trip: PASS")
    vna.close()

    # -- full VNACalibSlopeCustomMeas script, db + h5 export -------------

    from measurement_scripts.vna_calib_slope_custom_meas import VNACalibSlopeCustomMeas

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        circuit_path = tmp / "circuit.txt"
        circuit_path.write_text("fake circuit\n")
        save_path = str(tmp / "calib_sample1.h5")

        vna_params = {
            "visa_address": "TCPIP::169.254.1.3::INSTR",
            "nickname": "platoVNA",
            "config": {
                "power": -40.0,
                "measurements": (
                    ("Sig1Sig1", "S41"),
                    ("Sig2Sig1", "S21"),
                    ("Sig1Sig2", "S43"),
                    ("Sig2Sig2", "S23"),
                ),
            },
        }
        sweep_params = {
            "pts_list": (11, 21),
            "bw_list": (500, 1000),
            "power_slope": 2.0,
        }

        with patched_visa({vna_params["visa_address"]: FakePNAResource()}):
            meas = VNACalibSlopeCustomMeas(save_path, str(circuit_path), vna_params, sweep_params)
            meas.execute()

        # -- top-level file: BaseMeasurement metadata, empty data group --
        with h5.File(save_path, "r") as f:
            assert set(f.keys()) == {"metadata", "data"}
            assert list(f["data"].keys()) == []
            assert f["metadata"]["measurement"]["meas_type"][()].decode().endswith(
                "VNACalibSlopeCustomMeas'>"
            )
            assert (
                f["metadata"]["instruments"]["platoVNA"]["visa_address"][()].decode()
                == vna_params["visa_address"]
            )
        logger.info("BaseMeasurement metadata block: PASS")

        # -- per-iteration exported files, legacy layout ----------------
        for n_pt in sweep_params["pts_list"]:
            file_path = save_path[:-3] + f"_{n_pt}pts.h5"
            with h5.File(file_path, "r") as f:
                assert list(f.keys()) == ["data"]
                freq = f["data"]["Vna frequencies (Hz)"][()]
                data = f["data"]["data"][()]
                assert freq.shape == (n_pt,)
                assert data.shape == (n_pt, 2, 2)
                assert data.dtype == complex
                assert f["data"]["data"].dims[0].label == "Vna frequencies (Hz)"
                assert f["data"]["data"].dims[1].label == "X in SigXSigY"
                assert f["data"]["data"].dims[2].label == "Y in SigXSigY"
        logger.info("Exported per-iteration .h5 files: PASS")

        # -- qcodes db: one run per (pts, bw) pair, arrays round-trip ----
        db_path = tmp / "experiments.db"
        assert db_path.exists()
        from qcodes.dataset import initialise_or_create_database_at
        from qcodes.dataset.data_set import load_by_id

        initialise_or_create_database_at(db_path)
        for run_id, n_pt in zip((1, 2), sweep_params["pts_list"]):
            ds = load_by_id(run_id)
            pdata = ds.get_parameter_data()
            assert pdata["Sig1Sig1"]["Sig1Sig1"].size == n_pt
            assert ds.metadata["n_pts"] == str(n_pt)
        logger.info("QCoDeS .db runs: PASS")

    # -- full SpectroDCSweepSlope script, db + h5 export ------------------
    from measurement_scripts.spectro_flux_sweep_slope import SpectroDCSweepSlope

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        circuit_path = tmp / "circuit_can_1.txt"
        circuit_path.write_text("fake circuit\n")
        save_path = str(tmp / "flux_map.h5")

        vna_address = "TCPIP::169.254.1.3::INSTR"
        yoko_address = "TCPIP::169.254.1.4::INSTR"
        vna_params = {
            "visa_address": vna_address,
            "nickname": "platoVNA",
            "config": {"power": -40.0, "if_bandwidth": 500, "freq_spec": (3e9, 13e9, 21)},
        }
        yoko_params = {
            "visa_address": yoko_address,
            "nickname": "Yoko_Quantic2",
            "config": {
                "mode": "CURR",
                "current_value": 0,
                "current_range": "1 mA",
                "output": "on",
            },
        }
        sweep_params = {
            "current_start": -400e-6,
            "current_end": 400e-6,
            "n_current": 5,
            "power_slope": 2,
        }

        yoko_resource = FakeYokogawaResource()
        with patched_visa({vna_address: FakePNAResource(), yoko_address: yoko_resource}):
            meas = SpectroDCSweepSlope(save_path, str(circuit_path), vna_params, yoko_params, sweep_params)
            meas.execute()
        # execute()'s safe_run() closes both instruments on the way out, so
        # by this point meas.yoko is dead - check the actual wire traffic
        # instead: the output must have been switched off *after* the last
        # time it was switched on (both by reset_config() and by
        # safe_run()'s safe_shutdown(), even though nothing failed here).
        output_writes = [c for c in yoko_resource.write_log if c.upper().startswith("OUTPUT ")]
        assert output_writes[-1].upper() == "OUTPUT 0", output_writes
        assert output_writes.count("OUTPUT 1") >= 1  # config turned it on for the sweep
        logger.info("Yokogawa left in a safe (output-off) state after execute(): PASS")

        n_current = sweep_params["n_current"]
        n_freq = vna_params["config"]["freq_spec"][2]
        vna_meas = ("Sig1Sig1", "Sig2Sig1", "Sig1Sig2", "Sig2Sig2")

        with h5.File(save_path, "r") as f:
            assert set(f.keys()) == {"metadata", "data"}
            data_group = f["data"]
            assert set(data_group.keys()) == {"vna frequencies"} | {
                str(i).zfill(len(str(n_current))) for i in range(1, n_current + 1)
            }
            assert data_group["vna frequencies"].shape == (n_freq,)
            for i in range(1, n_current + 1):
                dset = data_group[str(i).zfill(len(str(n_current)))]
                assert dset.shape == (n_freq, len(vna_meas))
                assert dset.dtype == complex
                assert "time" in dset.attrs
                assert "Current (mA)" in dset.attrs
                for i_meas, label in enumerate(vna_meas):
                    assert dset.attrs[f"Data col {i_meas}"] == label
        logger.info("SpectroDCSweepSlope exported .h5 (single-file, per-current datasets): PASS")

        db_path = tmp / "experiments.db"
        from qcodes.dataset import initialise_or_create_database_at
        from qcodes.dataset.data_set import load_by_id

        initialise_or_create_database_at(db_path)
        ds = load_by_id(1)
        pdata = ds.get_parameter_data()
        assert pdata["Sig1Sig1"]["Sig1Sig1"].shape == (n_current, n_freq)
        assert "instruments" in ds.metadata
        logger.info("SpectroDCSweepSlope QCoDeS .db run: PASS")

    # -- full TwoToneSpectro script, db + h5 export ------------------------
    # AnaPicoAPUASYN20 doesn't go through qcodes' VisaInstrument at all - it
    # wraps the exopy-legacy Anapico1 driver, which opens its own
    # connection via legacy.drivers.exopy_hqc_legacy.drivers.visa_tools
    # .ResourceManager. Reuse the exact fake already built and verified for
    # that dialect in verify_without_hardware.py rather than duplicating it.
    from verify_without_hardware import FakeAnapicoResource, _FakeResourceManager

    from measurement_scripts.two_tone_spectro import TwoToneSpectro

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        circuit_path = tmp / "circuit_can_1_plasma_2tone.txt"
        circuit_path.write_text("fake circuit\n")
        save_path = str(tmp / "2tone.h5")

        vna_address = "TCPIP::169.254.1.3::INSTR"
        yoko_address = "TCPIP::169.254.1.4::INSTR"
        vna_params = {
            "visa_address": vna_address,
            "nickname": "platoVNA",
            "config": {
                "power": -20.0,
                "if_bandwidth": 500,
                "freq_spec": (1e9, 1),
                "measurements": (("Sig2Sig1", "S21"),),
            },
        }
        pico_params = {"visa_address": "TCPIP::169.254.1.5::INSTR", "nickname": "platoPico"}
        yoko_params = {
            "visa_address": yoko_address,
            "nickname": "Yoko_Quantic2",
            "config": {"mode": "CURR", "current_value": 0, "current_range": "1 mA", "output": "on"},
        }

        n_current, n_pico_freq = 3, 4
        currents = np.linspace(-3e-5, 3e-5, n_current)
        vna_freqs = np.linspace(8.5e9, 8.0e9, n_current)
        pico_freqs = np.linspace(0.1e9, 20e9, n_pico_freq)
        pico_powers = np.clip(-30 + 2 * pico_freqs * 1e-9, -10, 23)
        sweep_params = {
            "pico_freqs": pico_freqs,
            "currents": currents,
            "vna_freqs": vna_freqs,
            "pico_powers": pico_powers,
        }

        with (
            patched_visa({vna_address: FakePNAResource(), yoko_address: FakeYokogawaResource()}),
            patch(
                "legacy.drivers.exopy_hqc_legacy.drivers.visa_tools.ResourceManager",
                lambda *a, **kw: _FakeResourceManager(FakeAnapicoResource),
            ),
        ):
            meas = TwoToneSpectro(
                save_path, str(circuit_path), vna_params, pico_params, yoko_params, sweep_params
            )
            meas.execute()

        with h5.File(save_path, "r") as f:
            data_group = f["data"]
            assert data_group["Vna frequencies (Hz)"].shape == (n_current,)
            assert data_group["Currents (A)"].shape == (n_current,)
            assert data_group["Anapico frequencies (Hz)"].shape == (n_pico_freq,)
            assert data_group["Anapico powers (dBm)"].shape == (n_pico_freq,)
            dset = data_group["data"]
            assert dset.shape == (n_current, n_pico_freq, 1)  # freq_spec=(1e9, 1) -> 1 CW point
            assert dset.dtype == complex
            assert dset.dims[0].label == "Currents (A)"
            assert dset.dims[1].label == "Anapico frequency (Hz)"
            assert dset.dims[2].label == "CW Time (s)"
            # every point's timestamp is kept now, not just the last one
            # (can't assert they're all distinct - the fake sweep runs
            # fast enough that several points can land in the same
            # second, and timestamps only have 1s resolution)
            time_dset = data_group["time"]
            assert time_dset.shape == (n_current, n_pico_freq)
            assert all(t for row in time_dset[()] for t in row)
        logger.info("TwoToneSpectro exported .h5 (single shared 3-D dataset): PASS")

        db_path = tmp / "experiments.db"
        from qcodes.dataset import initialise_or_create_database_at
        from qcodes.dataset.data_set import load_by_id

        initialise_or_create_database_at(db_path)
        ds = load_by_id(1)
        pdata = ds.get_parameter_data()
        assert pdata["Sig2Sig1"]["Sig2Sig1"].shape == (n_current * n_pico_freq, 1)
        logger.info("TwoToneSpectro QCoDeS .db run: PASS")

    # -- full SpectroAWGPumpSweep... script, db + h5 export, incl. the ------
    # -- sub-75mV FIR-compensation path ------------------------------------
    from measurement_scripts.spectro_awgPump_sweep_variable_ranges_simpNOCompOnly_powSlope_sweep_flux import (
        SpectroAWGPumpSweepFIRSimpNOCompSweepFlux,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        circuit_path = tmp / "circuit_can_1.txt"
        circuit_path.write_text("fake circuit\n")
        save_path = str(tmp / "awg_pump_sweep.h5")

        vna_address = "TCPIP::169.254.1.3::INSTR"
        yoko_address = "TCPIP::169.254.1.4::INSTR"
        awg_address = "TCPIP0::169.254.1.6::inst0::INSTR"  # full VISA string - native.instruments.KeysightM8195A, not the old bare-IP pyarbtools convention
        vna_params = {
            "visa_address": vna_address,
            "nickname": "platoVNA",
            "config": {"power": -40.0, "if_bandwidth": 500, "freq_spec": (3e9, 13e9, 5)},
        }
        awg_params = {"visa_address": awg_address, "nickname": "bigBoiAWG"}
        yoko_params = {
            "visa_address": yoko_address,
            "nickname": "Yoko_Quantic2",
            "config": {"mode": "CURR", "current_value": 0, "current_range": "1 mA", "output": "on"},
        }
        n_current, n_freq, n_main_amp = 2, 2, 3
        min_amp = 75e-3
        sweep_params = {
            "main_channel": 1,
            "freqs": [5e9, 6e9],
            "main_amp_starts": [50e-3] * n_freq,  # below the 75mV floor - exercises FIR compensation
            "main_amp_ends": [150e-3] * n_freq,
            "n_main_amp": n_main_amp,
            "power_slope": 1.5,
            "current_start": -1e-4,
            "current_end": 1e-4,
            "n_current": n_current,
        }

        awg_resource = FakeM8195AResourceExt()
        with (
            patched_visa({vna_address: FakePNAResource(), yoko_address: FakeYokogawaResource()}),
            patch(
                "pyvisa.ResourceManager",
                lambda *a, **kw: _FakeResourceManager(lambda: awg_resource),
            ),
        ):
            meas = SpectroAWGPumpSweepFIRSimpNOCompSweepFlux(
                save_path, str(circuit_path), vna_params, awg_params, yoko_params, sweep_params
            )
            meas.execute()

        # -- FIR compensation actually happened, only on the transitions ---
        # No write in _setup_instruments() (left untouched on purpose).
        # Per (current, freq) sub-run - amps [0.05, 0.10, 0.15] - exactly 2
        # writes: entering compensation at 0.05, leaving it once at 0.10;
        # 0.15 needs no write at all (already disengaged). x 4 sub-runs.
        fir_writes = [
            float(m.group(1))
            for cmd in awg_resource.write_log
            if (m := re.match(r"outp1:filt:\w+:scal ([\-0-9.eE]+)", cmd.strip(), re.I))
        ]
        expected_fir_scale = [50e-3 / min_amp, 1.0] * (n_current * n_freq)
        assert len(fir_writes) == len(expected_fir_scale), fir_writes
        for actual, expected in zip(fir_writes, expected_fir_scale):
            assert abs(actual - expected) < 1e-9, (fir_writes, expected_fir_scale)
        logger.info("AWG FIR-scale compensation sequence (edge-triggered, not per-point): PASS")

        n_freq_vna = vna_params["config"]["freq_spec"][2]
        with h5.File(save_path, "r") as f:
            data_group = f["data"]
            assert data_group["Vna frequencies (Hz)"].shape == (n_freq_vna,)
            assert data_group["Pump frequencies (Hz)"].shape == (n_freq,)
            assert data_group["Main pump amps (Vpk-pk)"].shape == (n_freq, n_main_amp)
            assert data_group["DC current (A)"].shape == (n_current,)

        for id_current in range(n_current):
            for id_freq in range(n_freq):
                file_path = save_path[:-3] + f"_current{id_current}_freq{id_freq}.h5"
                with h5.File(file_path, "r") as f:
                    dset = f["data"]
                    assert dset.shape == (n_main_amp, n_freq_vna, 2, 2)
                    assert dset.dtype == complex
                    # every requested amplitude is now actually measured -
                    # none skipped, including the sub-75mV one (index 0)
                    assert not np.all(dset[0] == 0)
                    assert not np.all(dset[1] == 0)
                    assert not np.all(dset[2] == 0)
                    assert "time" in dset.attrs
                    assert "skipped_main_amp_indices" not in dset.attrs
        logger.info("SpectroAWGPumpSweep exported per-(current,freq) .h5 files, none skipped: PASS")

        db_path = tmp / "experiments.db"
        from qcodes.dataset import initialise_or_create_database_at
        from qcodes.dataset.data_set import load_by_id

        initialise_or_create_database_at(db_path)
        # one shared run for the whole sweep (current, pump_freq, main_amp
        # all independent params) - not one run per (current, freq) pair;
        # that split only ever applied to the .h5 side (see above).
        ds = load_by_id(1)
        pdata = ds.get_parameter_data()
        # all 3 requested amplitudes actually get measured now (0 skipped)
        assert pdata["Sig1Sig1"]["Sig1Sig1"].shape == (n_current * n_freq * n_main_amp, n_freq_vna)
        assert "skipped_main_amps_below_75mV" not in ds.metadata
        logger.info("SpectroAWGPumpSweep QCoDeS .db run: PASS")


_CRASH_SUBPROCESS_SCRIPT = """
import sys
sys.path.insert(0, {repo_root!r})
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests_dir!r})

from verify_measurement_scripts import FakePNAResource, FakeYokogawaResource, patched_visa
from measurement_scripts.spectro_flux_sweep_slope import SpectroDCSweepSlope

vna_address = "TCPIP::169.254.1.3::INSTR"
yoko_address = "TCPIP::169.254.1.4::INSTR"
vna_params = {{
    "visa_address": vna_address,
    "nickname": "platoVNA",
    "config": {{"power": -40.0, "if_bandwidth": 500, "freq_spec": (3e9, 13e9, 5)}},
}}
yoko_params = {{
    "visa_address": yoko_address,
    "nickname": "Yoko_Quantic2",
    "config": {{"mode": "CURR", "current_value": 0, "current_range": "1 mA", "output": "on"}},
}}
sweep_params = {{
    "current_start": -4e-4,
    "current_end": 4e-4,
    "n_current": 30,
    "power_slope": 2,
    "flush_interval": 1.0,
}}

with patched_visa({{
    vna_address: FakePNAResource(),
    yoko_address: FakeYokogawaResource(point_delay=0.3),
}}):
    meas = SpectroDCSweepSlope(
        {save_path!r}, {circuit_path!r}, vna_params, yoko_params, sweep_params
    )
    meas.execute()
"""


def test_crash_durability() -> None:
    """The whole point of writing each point straight into the open `.h5`
    (see `SpectroDCSweepSlope.execute`) instead of exporting once at the
    end: a crash mid-sweep should still leave a valid, partially-filled
    `.h5` on disk. Prove it for real - launch the sweep in a subprocess,
    SIGKILL it partway through (no graceful shutdown at all, the closest
    thing to power loss short of actually cutting power), and check what
    survives.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        circuit_path = tmp / "circuit_can_1.txt"
        circuit_path.write_text("fake circuit\n")
        save_path = str(tmp / "flux_map_crash.h5")

        script_path = tmp / "run_sweep.py"
        script_path.write_text(
            _CRASH_SUBPROCESS_SCRIPT.format(
                repo_root=str(REPO_ROOT),
                src=str(SRC),
                tests_dir=str(Path(__file__).resolve().parent),
                save_path=save_path,
                circuit_path=str(circuit_path),
            )
        )

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # n_current=30 * point_delay=0.3s = 9s of loop time, plus a couple
        # seconds of qcodes/instrument construction overhead first; with
        # flush_interval=1.0s several real flushes happen well before
        # that. Killing at t=6s guarantees multiple flushes already
        # landed, while the sweep (needs ~11-12s total) is still well
        # short of completing - proving this isn't just reading the
        # final, cleanly-closed file.
        time.sleep(6)
        proc.kill()  # SIGKILL - no __exit__/finally in the child runs at all
        proc.wait(timeout=10)

        assert Path(save_path).exists(), "no .h5 was ever created before the kill"
        with h5.File(save_path, "r") as f:
            assert "metadata" in f  # BaseMeasurement's block, written before the sweep
            names = sorted(n for n in f["data"].keys() if n != "vna frequencies")
            n_recovered = len(names)
            logger.info("recovered %d/30 points after SIGKILL mid-sweep", n_recovered)
            assert 0 < n_recovered < 30, (
                "expected a partial file (some points recovered, sweep not "
                f"finished) - got {n_recovered}/30"
            )
            # spot-check the recovered points are real, complete data, not
            # truncated/corrupt entries
            first, last = names[0], names[-1]
            assert f["data"][first].shape == f["data"][last].shape
            assert not np.all(f["data"][last][()] == 0)
            assert "time" in f["data"][last].attrs
    logger.info("Crash durability (SIGKILL mid-sweep, partial .h5 recovered): PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
    test_crash_durability()
