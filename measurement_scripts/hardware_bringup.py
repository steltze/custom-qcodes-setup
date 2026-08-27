# %% [markdown]
# # Hardware bring-up / smoke test
#
# Run this **cell by cell** (VS Code Python extension: `Shift+Enter`, or the
# "Run Cell" links that appear above each `# %%` marker - no `.ipynb` needed,
# this runs as a Jupyter-style interactive session directly on this `.py`
# file). Nothing here runs unless you explicitly run its cell.
#
# Order matters and is deliberate:
# 1. Each instrument gets connected and exercised **on its own** first -
#    read-only checks before any write, one small safe write, then close.
#    A bad address on one instrument should never block you from checking
#    the others, which is why this doesn't go through the measurement
#    classes yet (they construct every instrument they need all at once in
#    `__init__`).
# 2. Only once all four are individually verified do the last four sections
#    construct the real measurement classes (`VNACalibSlopeCustomMeas`,
#    `SpectroDCSweepSlope`, `TwoToneSpectro`,
#    `SpectroAWGPumpSweepFIRSimpNOCompSweepFlux`) with deliberately tiny
#    `sweep_params` (a handful of points, not a real multi-hour sweep), run
#    them for real, and then open the resulting `.h5`/`.db` with
#    `inspect_h5`/`inspect_latest_run` so you can actually look at what got
#    written, not just check that nothing crashed.
#
# This complements, not replaces, `tests/verify_measurement_scripts.py`
# (same classes, fake SCPI, proves the Python plumbing) and stepping through
# a real run with breakpoints (best once you already trust the pieces below
# and want to watch one specific run happen).
#
# Fill in the real VISA addresses in the Config cell before running anything
# past it. Every write-side cell (setting a current, playing a waveform,
# enabling RF) has a comment right above it - read it before running.

# %%
import logging
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import h5py
import numpy as np
from qcodes.dataset import experiments, initialise_or_create_database_at

from legacy.instruments.AnaPicoAPUASYN20X import AnaPicoAPUASYN20X
from legacy.instruments.KeysightM8195A import KeysightM8195A
from stock_instruments.KeysightP5024A import KeysightP5024A
from stock_instruments.YokogawaGS200 import YokogawaGS200

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BRINGUP_DIR = REPO_ROOT / "data" / "bringup"
BRINGUP_DIR.mkdir(parents=True, exist_ok=True)

CIRCUIT_PATH = BRINGUP_DIR / "circuit.txt"
if not CIRCUIT_PATH.exists():
    CIRCUIT_PATH.write_text("hardware bring-up smoke test - not a real circuit\n")


def scratch_h5_path(name: str) -> str:
    """A fresh .h5 path each call - BaseMeasurement refuses to overwrite an
    existing one, so rerunning a smoke-test cell needs a new name, not a
    fixed one you'd have to delete by hand each time."""
    return str(BRINGUP_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.h5")


def inspect_h5(path: str) -> None:
    """Log every group/dataset (shape, dtype) and every attr in the file,
    so you can actually look at what a smoke test wrote."""
    logger.info("--- %s ---", path)
    with h5py.File(path, "r") as f:
        for key, val in f.attrs.items():
            logger.info("  @%s = %r", key, val)

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                logger.info("  %s: shape=%s dtype=%s", name, obj.shape, obj.dtype)
            else:
                logger.info("  %s/", name)
            for key, val in obj.attrs.items():
                shown = val if np.ndim(val) == 0 else f"array{np.shape(val)}"
                logger.info("    @%s = %r", key, shown)

        f.visititems(show)


def inspect_latest_run(db_path, experiment_name: str) -> None:
    """Open the most recent qcodes run for `experiment_name` in `db_path`
    and log its shape/metadata - "most recent" so rerunning a smoke-test
    cell doesn't require tracking run ids by hand."""
    initialise_or_create_database_at(db_path)
    matches = [e for e in experiments() if e.name == experiment_name]
    if not matches:
        logger.info("--- no experiment named %r in %s yet ---", experiment_name, db_path)
        return
    ds = matches[-1].data_sets()[-1]
    logger.info("--- %s :: %s :: run id %s ---", db_path, experiment_name, ds.run_id)
    logger.info("  completed: %s", ds.completed)
    pdata = ds.get_parameter_data()
    for dependent, columns in pdata.items():
        shapes = {name: arr.shape for name, arr in columns.items()}
        logger.info("  %s: %s", dependent, shapes)
    logger.info("  metadata keys: %s", list(ds.metadata.keys()))


logger.info("Setup OK. Scratch data directory: %s", BRINGUP_DIR)

# %% [markdown]
# ## Config - fill in your real VISA addresses before running anything below

# %%
VNA_ADDRESS = "TCPIP0::CHANGE-ME::inst0::INSTR"  # Network Analyzer app must already be running
YOKO_ADDRESS = "GPIB0::CHANGE-ME::INSTR"
AWG_ADDRESS = "TCPIP0::CHANGE-ME::inst0::INSTR"  # M8195 Soft Front Panel SCPI server
ANAPICO_ADDRESS = "TCPIP0::CHANGE-ME::inst0::INSTR"

SAFE_VNA_POWER = -40.0  # dBm - matches every real sweep_params in this repo
SAFE_AWG_AMP = 0.075  # V, the AWG's hardware floor (see _MIN_AMP in the AWG-pump script)

# %% [markdown]
# # 1. VNA (KeysightP5024A)
#
# Reads first, one small write, then close. `KeysightP5024A(...)` already
# presets the instrument and turns RF output *on* as part of construction
# (see `reset_config()`) - that's the point where output actually turns on,
# not a separate step below.

# %%
vna = KeysightP5024A(
    "vna_bringup",
    VNA_ADDRESS,
    config={"power": SAFE_VNA_POWER, "if_bandwidth": 1000, "freq_spec": (3e9, 13e9, 51)},
)
logger.info("%s", vna.get_idn())

# %%
# Read-only - confirm what actually got configured.
logger.info("power: %s", vna.power())
logger.info("if_bandwidth: %s", vna.if_bandwidth())
logger.info("points: %s", vna.points())
logger.info("output: %s", vna.output())
logger.info("power_slope: %s", vna.power_slope())

# %%
# One measurement, one averaged sweep, read it back.
vna.configure_measurements((("Sig2Sig1", "S21"),))
vna.run_averaging()
freq_data = vna.read_freq_data()
trace = vna.read_raw_data("Sig2Sig1")
logger.info("freq_data: %s %s ...", freq_data.shape, freq_data[:3])
logger.info("trace: %s %s %s", trace.shape, trace.dtype, trace[:3])

# %%
# power_slope get/set round-trip.
vna.power_slope(1.0)
logger.info("power_slope after set: %s", vna.power_slope())
vna.power_slope(0.0)

# %%
vna.close()
logger.info("VNA closed.")

# %% [markdown]
# # 2. Yokogawa (YokogawaGS200)
#
# `reset_config()` already ran in `__init__`, so output starts off - confirm
# that before doing anything else.

# %%
yoko = YokogawaGS200(
    "yoko_bringup", YOKO_ADDRESS,
    config={"mode": "CURR", "current_range": "1 mA"},
)
logger.info("%s", yoko.get_idn())
logger.info("output right after construction (should be False): %s", yoko.output())

# %%
# Read-only.
logger.info("source_mode: %s", yoko.source_mode())
logger.info("current_range: %s", yoko.current_range())
logger.info("current: %s", yoko.current())

# %% SAFETY: this turns the DC source's output ON at 0 A. Check the physical
# setup is ready for that before running this cell.
yoko.output("on")
yoko.current(0.0)
logger.info("current after set: %s", yoko.current())

# %%
yoko.reset_config()
logger.info("output after reset_config (should be False): %s", yoko.output())
yoko.close()
logger.info("Yokogawa closed.")

# %% [markdown]
# # 3. AWG (KeysightM8195A)

# %%
awg = KeysightM8195A("awg_bringup", AWG_ADDRESS)
logger.info("%s", awg.get_idn())

# %%
# Read-only.
logger.info("dac_mode: %s", awg.dac_mode())
logger.info("sample_rate: %s", awg.sample_rate())
logger.info("amplitude_1: %s", awg.amplitude_1())

# %% SAFETY: this downloads and plays a low-amplitude (75mV) sine on
# channel 1. Check the physical setup is ready for that before running this
# cell.
seg_id = awg.send_sine(freq=5e8, phase=0.0, channel=1, amp=SAFE_AWG_AMP)
awg.ask_if_done()
logger.info("segment id: %s amplitude_1 now: %s", seg_id, awg.amplitude_1())

# %%
awg.stop(1)
awg.clear_all_wfm()
awg.close()
logger.info("AWG closed.")

# %% [markdown]
# # 4. Anapico (AnaPicoAPUASYN20X - 4 channels)
#
# `outputs` defaults to all-off in `config` below, same "starts safe" pattern
# as the VNA/Yoko sections - nothing turns RF on until the explicit SAFETY
# cell.

# %%
pico = AnaPicoAPUASYN20X(
    "pico_bringup",
    ANAPICO_ADDRESS,
    config={
        "frequencies": (5e9, 5e9, 5e9, 5e9),
        # -10.0 dBm, not -20.0 - confirmed on real hardware this
        # channel's power floor is -10 dBm (range -10..+23 dBm); -20.0
        # is silently ignored by the instrument, not rejected, so it's
        # worth getting this one right rather than relying on the
        # power_N parameters' vals= bounds to catch it (this config dict
        # is applied straight through the legacy Anapico4 driver at
        # construction time, bypassing those bounds entirely).
        "powers": (-10.0, -10.0, -10.0, -10.0),
        "outputs": ("OFF", "OFF", "OFF", "OFF"),
    },
)
logger.info("%s", pico.get_idn())

# %%
# Read-only, all 4 channels at once.
for ch in (1, 2, 3, 4):
    logger.info(
        "ch%d: freq=%s power=%s",
        ch, getattr(pico, f"frequency_{ch}")(), getattr(pico, f"power_{ch}")(),
    )
logger.info("which_outputs_enabled (should be all False): %s", pico.which_outputs_enabled())
logger.info("reference_source: %s", pico.reference_source())

# %% SAFETY: this turns RF output ON at -10 dBm, 5 GHz, on channel 1 only.
# Check the physical setup is ready for that before running this cell.
pico.output_enabled_1(True)
logger.info("which_outputs_enabled after set: %s", pico.which_outputs_enabled())

# %%
pico.safe_shutdown()
logger.info("which_outputs_enabled after safe_shutdown (should be all False): %s", pico.which_outputs_enabled())
pico.close()
logger.info("Anapico closed.")

# %% [markdown]
# # 5. VNACalibSlopeCustomMeas - VNA only
#
# Real class, real `execute()`, tiny `sweep_params` (one 11-point sweep
# instead of the usual `(10001,), (2501,), (2501,)`).

# %%
from measurement_scripts.vna_calib_slope_custom_meas import VNACalibSlopeCustomMeas  # noqa: E402

vna_calib_save_path = scratch_h5_path("vna_calib")
vna_calib = VNACalibSlopeCustomMeas(
    vna_calib_save_path,
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
vna_calib.execute()
logger.info("done: %s", vna_calib_save_path)

# %%
inspect_h5(vna_calib_save_path)
inspect_h5(vna_calib_save_path[:-3] + "_11pts.h5")
inspect_latest_run(BRINGUP_DIR / "experiments.db", "vna_calib_slope_custom_meas")

# %% [markdown]
# # 6. SpectroDCSweepSlope - VNA + Yokogawa
#
# Tiny current sweep (2 points instead of the usual ~501, right around 0 A).

# %%
from measurement_scripts.spectro_flux_sweep_slope import SpectroDCSweepSlope  # noqa: E402

flux_save_path = scratch_h5_path("spectro_flux")
flux_meas = SpectroDCSweepSlope(
    flux_save_path,
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
flux_meas.execute()
logger.info("done: %s", flux_save_path)

# %%
inspect_h5(flux_save_path)
inspect_latest_run(BRINGUP_DIR / "experiments.db", "spectro_flux_sweep_slope")

# %% [markdown]
# # 7. TwoToneSpectro - VNA + Anapico + Yokogawa
#
# Tiny 2x2 grid instead of the usual 17 currents x 4001 pico frequencies.

# %%
from measurement_scripts.two_tone_spectro import TwoToneSpectro  # noqa: E402

two_tone_save_path = scratch_h5_path("two_tone")
two_tone_meas = TwoToneSpectro(
    two_tone_save_path,
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
        "pico_powers": np.array([-10.0, -10.0]),  # -10 dBm floor - see Anapico section above
    },
)
two_tone_meas.execute()
logger.info("done: %s", two_tone_save_path)

# %%
inspect_h5(two_tone_save_path)
inspect_latest_run(BRINGUP_DIR / "experiments.db", "two_tone_spectro")

# %% [markdown]
# # 8. SpectroAWGPumpSweepFIRSimpNOCompSweepFlux - VNA + AWG + Yokogawa
#
# Tiny sweep: 1 current x 1 pump frequency x 2 amplitudes, both amplitudes
# at/just above the 75mV floor so nothing gets skipped.

# %%
from measurement_scripts.spectro_awgPump_sweep_variable_ranges_simpNOCompOnly_powSlope_sweep_flux import (  # noqa: E402
    SpectroAWGPumpSweepFIRSimpNOCompSweepFlux,
)

awg_pump_save_path = scratch_h5_path("awg_pump")
awg_pump_meas = SpectroAWGPumpSweepFIRSimpNOCompSweepFlux(
    awg_pump_save_path,
    str(CIRCUIT_PATH),
    vna_params={
        "visa_address": VNA_ADDRESS,
        "nickname": "vna_awgpump_bringup",
        "config": {"if_bandwidth": 1000, "freq_spec": (3e9, 13e9, 11)},
    },
    awg_params={"visa_address": AWG_ADDRESS, "nickname": "awg_awgpump_bringup"},
    dc_params={
        "visa_address": YOKO_ADDRESS,
        "nickname": "yoko_awgpump_bringup",
        "config": {"mode": "CURR", "current_value": 0, "current_range": "1 mA", "output": "on"},
    },
    sweep_params={
        "main_channel": 1,
        "freqs": [5e8],
        "main_amp_starts": [SAFE_AWG_AMP],
        "main_amp_ends": [0.1],
        "n_main_amp": 2,
        "power_slope": 0.0,
        "current_start": 0.0,
        "current_end": 0.0,
        "n_current": 1,
    },
)
awg_pump_meas.execute()
logger.info("done: %s", awg_pump_save_path)

# %%
inspect_h5(awg_pump_save_path)
inspect_h5(awg_pump_save_path[:-3] + "_current0_freq0.h5")
inspect_latest_run(BRINGUP_DIR / "experiments.db", "spectro_awgPump_sweep_flux")

# %% [markdown]
# # Done
#
# Every instrument's `safe_shutdown()` already ran at the end of each
# `execute()` above (via `safe_run` - see `qcodes_utils/measurement_run.py`),
# so nothing needs manual closing here. `data/bringup/` now holds every
# `.h5`/`.db` this produced - delete the directory once you're done
# reviewing it, it's just scratch output (already covered by the repo's
# `.gitignore`, which ignores `*.h5`/`*.db`).
