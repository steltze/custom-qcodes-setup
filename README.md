# custom-qcodes-setup

Custom QCoDeS instrument drivers and measurement scripts for the lab setup.

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
