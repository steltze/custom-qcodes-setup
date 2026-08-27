#!/usr/bin/env python3
"""AWG-pumped spectroscopy, swept over pump frequency/amplitude and flux
current (slope-compensated, no gain compensation).

Fill in the VISA addresses in the CONFIG section below, then run in the
background with (Linux):

    nohup python examples/spectro_awg.py > examples/spectro_awg.out 2>&1 &

See the README's "Running unattended" section for the Windows equivalent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qcodes.instrument import Instrument

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from examples._common import make_save_file, require_addresses, setup_logging
from measurement_scripts.spectro_awgPump_sweep_variable_ranges_simpNOCompOnly_powSlope_sweep_flux import (
    SpectroAWGPumpSweepFIRSimpNOCompSweepFlux,
)

logger = setup_logging()

# --- CONFIG: fill in before running ------------------------------------
VNA_ADDRESS = ""  # e.g. "TCPIP0::192.168.1.5::inst0::INSTR"
YOKO_ADDRESS = ""  # e.g. "GPIB0::5::INSTR"
AWG_ADDRESS = ""  # e.g. "TCPIP0::192.168.1.7::inst0::INSTR"

SAVE_DIR = Path("./data")
CIRCUIT_PATH = SAVE_DIR / "circuit.txt"  # must already exist - see BaseMeasurement

PUMP_FREQS = np.linspace(1e9, 7e9, 15)
MAIN_AMP_STARTS = np.full_like(PUMP_FREQS, 25e-3)
MAIN_AMP_ENDS = np.full_like(PUMP_FREQS, 500e-3)
CURRENT_START = 0.08e-3
CURRENT_END = 0.12e-3
N_CURRENT = 3
# ------------------------------------------------------------------------


def main() -> None:
    require_addresses(VNA_ADDRESS=VNA_ADDRESS, YOKO_ADDRESS=YOKO_ADDRESS, AWG_ADDRESS=AWG_ADDRESS)

    logger.info("Closing any already-open QCoDeS instruments before starting")
    Instrument.close_all()

    save_file_path = make_save_file(SAVE_DIR, "awg_pump")

    vna_params = {
        "visa_address": VNA_ADDRESS,
        "nickname": "platoVNA",
        "config": {"power": -50.0, "if_bandwidth": 1000, "freq_spec": (3e9, 13e9, 300)},
    }
    pump_params = {"visa_address": AWG_ADDRESS, "nickname": "bigBoiAWG"}
    yoko_params = {
        "visa_address": YOKO_ADDRESS,
        "nickname": "Yoko_Quantic2",
        "config": {
            "mode": "CURR",
            "current_value": CURRENT_START,
            "current_range": "1 mA",
            "output": "on",
        },
    }
    sweep_params = {
        "main_channel": 1,
        "freqs": PUMP_FREQS,
        "main_amp_starts": MAIN_AMP_STARTS,
        "main_amp_ends": MAIN_AMP_ENDS,
        "n_main_amp": 10,
        "power_slope": 2.0,
        "current_start": CURRENT_START,
        "current_end": CURRENT_END,
        "n_current": N_CURRENT,
    }

    awg_pump_meas = SpectroAWGPumpSweepFIRSimpNOCompSweepFlux(
        save_file_path, str(CIRCUIT_PATH), vna_params, pump_params, yoko_params, sweep_params
    )
    logger.info("Starting AWG-pumped spectroscopy sweep, saving to %s", save_file_path)
    awg_pump_meas.execute()
    logger.info("Done: %s", save_file_path)


if __name__ == "__main__":
    main()
