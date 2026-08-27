"""Shared helpers for the example measurement scripts in this directory.

Not part of the installed package. Each script must put the repo root on
`sys.path` itself, *before* importing this module - `python examples/x.py`
puts `examples/` (not the repo root) on `sys.path[0]`, so this module can't
bootstrap that path for its own import (chicken-and-egg). See any script in
this directory for the exact bootstrap snippet.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_logging() -> logging.Logger:
    """Console + timestamped-file logging via qcodes' own logger
    (`qcodes.logger.start_all_logging`), redirected from its ~/.qcodes/logs/
    default to data/logs/ so a run's log sits next to the data it produced.
    Attaches to the root logger, so every plain `logging.getLogger(__name__)`
    call elsewhere in this repo is captured too - not just qcodes' own.

    Must run before anything else touches qcodes logging; QCODES_USER_PATH
    is only read at call time, so setting it here first is enough."""
    from qcodes.logger import start_all_logging

    os.environ.setdefault("QCODES_USER_PATH", str(REPO_ROOT / "data"))
    start_all_logging()
    return logging.getLogger(__name__)


def require_addresses(**addresses: str) -> None:
    """Fail immediately if a VISA address placeholder was left blank, instead
    of failing confusingly (or hanging) once a real instrument connection is
    attempted - important for a script left running unattended."""
    missing = [name for name, address in addresses.items() if not address]
    if missing:
        raise ValueError(
            f"Set {', '.join(missing)} in the CONFIG section before running "
            "this script."
        )


def make_save_file(save_dir: Path, prefix: str) -> Path:
    """A fresh, timestamped .h5 path under save_dir, creating save_dir if
    needed. BaseMeasurement refuses to overwrite an existing file, so a
    timestamp keeps repeated runs from colliding."""
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.h5"
