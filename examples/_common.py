"""Shared helpers for the example measurement scripts in this directory.

Not part of the installed package. Each script must put the repo root on
`sys.path` itself, *before* importing this module - `python examples/x.py`
puts `examples/` (not the repo root) on `sys.path[0]`, so this module can't
bootstrap that path for its own import (chicken-and-egg). See any script in
this directory for the exact bootstrap snippet.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Timestamped logging to stdout, so progress is visible in a redirected
    nohup log file when checked on later."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
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
