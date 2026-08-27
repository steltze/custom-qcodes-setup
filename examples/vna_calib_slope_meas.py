#!/usr/bin/env python3
"""VNA calibration sweep (slope-compensated), custom measurement.

Fill in VNA_ADDRESS in the CONFIG section below, then run in the
background with (Linux):

    nohup python examples/vna_calib_slope_meas.py > examples/vna_calib_slope_meas.out 2>&1 &

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
from measurement_scripts.vna_calib_slope_custom_meas import VNACalibSlopeCustomMeas

logger = setup_logging()

# --- CONFIG: fill in before running ------------------------------------
VNA_ADDRESS = ""  # e.g. "TCPIP0::192.168.1.5::inst0::INSTR"
SAFE_VNA_POWER = -40

SAVE_DIR = Path("./data")
CIRCUIT_PATH = SAVE_DIR / "circuit.txt"  # must already exist - see BaseMeasurement
# ------------------------------------------------------------------------


def main() -> None:
    require_addresses(VNA_ADDRESS=VNA_ADDRESS)
    save_file_path = make_save_file(SAVE_DIR, "vna_calib")

    vna_calib = VNACalibSlopeCustomMeas(
        save_file_path,
        str(CIRCUIT_PATH),
        vna_params={
            "visa_address": VNA_ADDRESS,
            "nickname": "vna_calib_bringup",
            "config": {
                "power": SAFE_VNA_POWER,
                "freq_spec": (3e9, 13e9, 11),
                "measurements": (
                    ("Sig1Sig1", "S41"),
                    ("Sig2Sig1", "S21"),
                    ("Sig1Sig2", "S43"),
                    ("Sig2Sig2", "S23"),
                ),
            },
        },
        sweep_params={"pts_list": (11,), "bw_list": (1000,), "power_slope": 0.0},
    )
    logger.info("Starting VNA calibration sweep, saving to %s", save_file_path)
    vna_calib.execute()
    logger.info("Done: %s", save_file_path)


if __name__ == "__main__":
    main()
