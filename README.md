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
