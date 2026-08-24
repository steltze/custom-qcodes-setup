"""Shared plumbing for measurement_scripts/*.py: opens the qcodes sqlite
database each run is captured into, and puts every instrument into a safe
state (then closes its connection) no matter how the run ends.

Kept separate from `measurement_scripts/base_measurement.py` on purpose:
`BaseMeasurement` reproduces the *old* codebase's per-run `.h5` metadata
block exactly, and every new measurement class still inherits from it for
byte-identical output there - this module is only the new, qcodes-specific
half: the sqlite `.db` and the instrument safety net.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from qcodes.dataset import initialise_or_create_database_at, load_or_create_experiment
from qcodes.dataset.experiment_container import Experiment
from qcodes.instrument import InstrumentBase

log = logging.getLogger(__name__)


def open_experiment(
    save_path: str, experiment_name: str, sample_name: str
) -> Experiment:
    """Point qcodes at a sqlite database next to `save_path` - one
    `experiments.db` shared by every run written into that data folder, the
    normal qcodes way (*not* one database per measurement, unlike the
    per-run legacy `.h5` files) - and return the experiment new runs
    should be added to.
    """
    db_path = Path(save_path).resolve().parent / "experiments.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    initialise_or_create_database_at(db_path)
    return load_or_create_experiment(
        experiment_name=experiment_name, sample_name=sample_name
    )


def instrument_metadata(
    instruments: Iterable[InstrumentBase],
    params: dict[str, Any],
    circuit_path: str,
) -> dict[str, str]:
    """The same information `BaseMeasurement.save_measurement_info` writes
    into the legacy `.h5` file's `metadata` group, shaped for
    `dataset.add_metadata` (which only accepts string values - dict/tuple/
    array values are JSON-encoded, with `default=str` as a fallback for
    numpy arrays)."""
    instruments_info = {}
    for instr in instruments:
        info = instr.get_config_info()
        instruments_info[info["nickname"]] = info
    return {
        "instruments": json.dumps(instruments_info, default=str),
        "measurement_params": json.dumps(params, default=str),
        "circuit_path": str(circuit_path),
    }


class PeriodicFlush:
    """Cheap time-based throttle for how often a sweep loop pushes data to
    disk, so a crash/power-loss mid-sweep loses at most `interval` seconds
    of the *already-written* `.h5`/`.db` - not the whole run.

    Deliberately dumb: every point is written into the already-open `.h5`
    file (and added to the qcodes datasaver) as soon as it's measured -
    that part is O(1) per point and never re-reads or re-holds earlier
    points in memory. This class only decides when it's worth also paying
    for an OS-level `flush()`/`flush_data_to_database()` call, so a fast
    sweep (thousands of points/second) doesn't fsync on every single
    point.

    Usage: call `due()` once per loop iteration, right after writing that
    iteration's point; when it returns True, flush the `.h5` file handle
    and call `datasaver.flush_data_to_database()`. Call both flushes once
    more, unconditionally, after the loop ends.
    """

    def __init__(self, interval: float = 60.0) -> None:
        """
        internval: in seconds
        """
        self.interval = interval
        self._last = time.monotonic()

    def due(self) -> bool:
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False


@contextlib.contextmanager
def safe_run(instruments: Iterable[InstrumentBase]):
    """Run the `with` block; whatever happens, drive every instrument to
    its safe state and close its connection afterwards.

    On the way out, every instrument's `safe_shutdown()` (if it defines
    one - RF/current output off, no waveform memory wiped) is called
    unconditionally - a clean finish as much as an exception or Ctrl-C,
    not just the abort case. Connections are always closed afterwards
    either way, so a run never leaves a stale VISA session behind.
    """
    instruments = list(instruments)
    try:
        yield
    finally:
        for instr in instruments:
            shutdown = getattr(instr, "safe_shutdown", None)
            if shutdown is None:
                continue
            try:
                shutdown()
            except Exception:
                log.exception("safe_shutdown() failed for %s", instr.name)
        for instr in instruments:
            try:
                instr.close()
            except Exception:
                log.exception("close() failed for %s", instr.name)
