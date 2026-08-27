#!/usr/bin/env python3
"""Spectroscopy vs. DC flux sweep (slope-compensated VNA power).

Fill in the VISA addresses in the CONFIG section below, then run in the
background with (Linux):

    nohup python examples/spectro_dc_sweep_slope.py > examples/spectro_dc_sweep_slope.out 2>&1 &

See the README's "Running unattended" section for the Windows equivalent.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from examples._common import make_save_file, require_addresses, setup_logging
from measurement_scripts.spectro_flux_sweep_slope import SpectroDCSweepSlope

logger = setup_logging()

# --- CONFIG: fill in before running ------------------------------------
VNA_ADDRESS = ""  # e.g. "TCPIP0::192.168.1.5::inst0::INSTR"
YOKO_ADDRESS = ""  # e.g. "GPIB0::5::INSTR"
SAFE_VNA_POWER = -40

SAVE_DIR = Path("./data")
CIRCUIT_PATH = SAVE_DIR / "circuit.txt"  # must already exist - see BaseMeasurement
# ------------------------------------------------------------------------


def main() -> None:
    require_addresses(VNA_ADDRESS=VNA_ADDRESS, YOKO_ADDRESS=YOKO_ADDRESS)
    save_file_path = make_save_file(SAVE_DIR, "spectro_dc_sweep_slope")

    flux_meas = SpectroDCSweepSlope(
        save_file_path,
        str(CIRCUIT_PATH),
        vna_params={
            "visa_address": VNA_ADDRESS,
            "nickname": "vna_flux_bringup",
            "config": {"power": SAFE_VNA_POWER, "if_bandwidth": 1000, "freq_spec": (3e9, 13e9, 11)},
        },
        dc_params={
            "visa_address": YOKO_ADDRESS,
            "nickname": "yoko_flux_bringup",
            "config": {"mode": "CURR", "current_value": 0, "current_range": "1 mA", "output": "on"},
        },
        sweep_params={
            "current_start": -1e-6,
            "current_end": 1e-6,
            "n_current": 2,
            "power_slope": 0.0,
        },
    )
    logger.info("Starting DC flux sweep, saving to %s", save_file_path)
    flux_meas.execute()
    logger.info("Done: %s", save_file_path)


if __name__ == "__main__":
    main()
