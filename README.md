# custom-qcodes-setup

QCoDeS instrument drivers and measurement scripts for LPENS-Quantic's
superconducting-circuit experiments.

> **Note:** This project is not intended to be a standalone library. Rather, it serves as a reference implementation and provides preliminary instrument drivers not yet supported by the official [QCoDeS](https://github.com/QCoDeS/Qcodes) distribution or its [community drivers](https://github.com/QCoDeS/Qcodes_contrib_drivers) repository.

## Setup

Dependencies and the virtual environment are managed with [uv](https://docs.astral.sh/uv/).
`uv.lock` pins exact, reproducible versions for both Windows and Linux.

1. Install uv (one-time, per machine):
   - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. From the repo root, create the environment and install the exact locked dependencies:
   ```
   uv sync
   ```
   This creates `.venv` (Linux: `.venv/bin/`, Windows: `.venv\Scripts\`) and installs everything from `uv.lock`.
3. Run scripts with `uv run`, e.g. `uv run python measurement_scripts/hardware_bringup.py`,
   or activate the environment directly (`source .venv/bin/activate` on Linux,
   `.venv\Scripts\activate` on Windows) and run `python ...` as usual.

To add a new dependency: `uv add <package>`, which updates `pyproject.toml` and `uv.lock` together.

For real hardware, VISA communication also needs a VISA backend on the machine
(e.g. NI-VISA) or `pyvisa-py`/the `sim` extra (`uv sync --extra sim`) for
simulated instruments in tests.

## Examples

`examples/` holds runnable, standalone measurement scripts (`vna_calib_slope_meas.py`,
`spectro_dc_sweep_slope.py`, `two_tone_spectro.py`, `spectro_awg.py`) built on the
classes in `measurement_scripts/`. Each one:

- fails fast with a clear error if a VISA address placeholder in its CONFIG
  section was left blank, instead of hanging or failing confusingly once a
  connection is attempted;
- logs progress with timestamps (`logging`, not `print`), so a run left
  unattended shows what happened and when if you check on it later;
- writes to a fresh, timestamped `.h5` file under `./data`, so repeated runs
  never collide.

Fill in the VISA addresses at the top of the script you want, then run it.

### Running unattended

To start a measurement and let it keep running after you log out or close
the terminal:

**Linux:**
```
nohup python examples/spectro_awg.py > examples/spectro_awg.out 2>&1 &
```
`nohup` stops the process from being killed when the terminal hangs up;
`&` backgrounds it; the redirect captures the timestamped log output to a
file you can `tail -f` later.

**Windows (PowerShell)** has no direct `nohup`, but `Start-Process` is the
equivalent - it detaches the process so it survives the console closing,
and can redirect output the same way:
```
Start-Process -NoNewWindow python -ArgumentList "examples\spectro_awg.py" `
  -RedirectStandardOutput examples\spectro_awg.out -RedirectStandardError examples\spectro_awg.err
```
For something that needs to keep running across a full logout (not just a
closed terminal), use Task Scheduler (`schtasks /create ...`) instead.

## Capability matrix

Instruments are covered by either a from-scratch native driver (`src/native/`,
no qcodes/exopy/pyarbtools dependency) or a thin wrapper around a stock
QCoDeS driver (`src/stock_instruments/`).

| Instrument | Native (`src/native/`) | Stock QCoDeS (`src/stock_instruments/`) | Status |
|---|---|---|---|
| AnaPico APUASYN20 (1-ch signal source) | based on SCPI/pyvisa | — | confirmed on hardware |
| AnaPico APUASYN20-X (4-ch signal source) | based on SCPI/pyvisa | — | confirmed on hardware |
| Keysight M8195A (65 GSa/s AWG) | based on SCPI/pyvisa | — | confirmed on hardware |
| Keysight P5024A (VNA) | — | QCoDeS native `KeysightPNAxBase` + config-dict wrapper | Fully on stock QCoDeS |
| Yokogawa GS200 (DC source) | — | QCoDeS native `YokogawaGS200` + config-dict wrapper | Fully on stock QCoDeS |
| Signal Hound SA124B (spectrum analyzer) | — | QCoDeS native, ctypes `sa_api` SDK (different transport, no VISA) | Fully on stock QCoDeS |

`tests/verify_native_drivers.py` exercises the native drivers against a
mocked VISA layer, without touching real instruments.
