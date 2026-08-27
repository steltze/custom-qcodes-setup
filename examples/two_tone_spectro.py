#!/usr/bin/env python3
"""Two-tone spectroscopy vs. DC flux.

Fill in the VISA addresses in the CONFIG section below, then run in the
background with (Linux):

    nohup python examples/two_tone_spectro.py > examples/two_tone_spectro.out 2>&1 &

See the README's "Running unattended" section for the Windows equivalent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from examples._common import make_save_file, require_addresses, setup_logging
from measurement_scripts.two_tone_spectro import TwoToneSpectro

logger = setup_logging()

# --- CONFIG: fill in before running ------------------------------------
VNA_ADDRESS = ""  # e.g. "TCPIP0::192.168.1.5::inst0::INSTR"
YOKO_ADDRESS = ""  # e.g. "GPIB0::5::INSTR"
ANAPICO_ADDRESS = ""  # e.g. "TCPIP0::192.168.1.6::inst0::INSTR"
SAFE_VNA_POWER = -40

SAVE_DIR = Path("./data")
CIRCUIT_PATH = SAVE_DIR / "circuit.txt"  # must already exist - see BaseMeasurement
# ------------------------------------------------------------------------


def main() -> None:
    require_addresses(
        VNA_ADDRESS=VNA_ADDRESS, YOKO_ADDRESS=YOKO_ADDRESS, ANAPICO_ADDRESS=ANAPICO_ADDRESS
    )
    save_file_path = make_save_file(SAVE_DIR, "two_tone_spectro")

    two_tone_meas = TwoToneSpectro(
        save_file_path,
        str(CIRCUIT_PATH),
        vna_params={
            "visa_address": VNA_ADDRESS,
            "nickname": "vna_2tone_bringup",
            "config": {
                "power": SAFE_VNA_POWER,
                "if_bandwidth": 1000,
                "freq_spec": (5e9, 1),  # (CW frequency, 1 point) - required 2-tuple, see class docstring
                "measurements": (("Sig2Sig1", "S21"),),
            },
        },
        pico_params={"visa_address": ANAPICO_ADDRESS, "nickname": "pico_2tone_bringup"},
        dc_params={
            "visa_address": YOKO_ADDRESS,
            "nickname": "yoko_2tone_bringup",
            "config": {"mode": "CURR", "current_value": 0, "current_range": "1 mA", "output": "on"},
        },
        sweep_params={
            "currents": np.array([0.0, 1e-6]),
            "vna_freqs": np.array([5e9, 5e9]),
            "pico_freqs": np.array([4e9, 6e9]),
            "pico_powers": np.array([-8.0, -9.0]),
        },
    )
    logger.info("Starting two-tone spectroscopy sweep, saving to %s", save_file_path)
    two_tone_meas.execute()
    logger.info("Done: %s", save_file_path)


if __name__ == "__main__":
    main()
