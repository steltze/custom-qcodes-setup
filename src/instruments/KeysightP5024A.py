from __future__ import annotations

import time
from typing import Any

import numpy as np

from qcodes.instrument_drivers.Keysight.N52xx import KeysightPNAxBase, KeysightPNATrace
from qcodes.parameters import Parameter


# ---------------------------------------------------------------------------
# Keysight P5024A Streamline VNA
# ---------------------------------------------------------------------------
class KeysightP5024A(KeysightPNAxBase):
    """
    Keysight P5024A Streamline USB VNA.

    The Streamline series runs the same firmware as the PNA family, so the
    stock ``KeysightPNAxBase`` does the heavy lifting (channel/trace model,
    sweep/average control, `points`/`if_bandwidth`/`cw`/`start`/`stop`/
    `power`, and per-trace `.polar()` which already returns complex I/Q
    data). This subclass adds:

    - the same ``config`` dict convenience
      ``instruments_old/vna_P5024A.py::VNA_P5024A`` had, so call sites
      barely change;
    - ``power_slope``/``power_slope_state`` (PNA-specific, confirmed absent
      from the native driver - see
      ``instruments_old/exopy_hqc_legacy/drivers/visa/agilent_pna.py``);
    - ``configure_measurements``/``read_raw_data``/``run_averaging``, thin
      conveniences on top of the native trace model matching the old
      driver's named-measurement vocabulary (``'Sig1Sig1' -> 'S23'`` etc).

    NOTE: the *Network Analyzer application must be running on the host PC*
    before any SCPI interface exists - the instrument itself is faceless.

    To confirm the limits from the instrument itself::

        inst.ask(':SENS:FREQ:STAR? MIN')
        inst.ask(':SENS:FREQ:STOP? MAX')
        inst.ask(':SOUR:POW? MIN')
        inst.ask(':SOUR:POW? MAX')
        inst.ask(':SYST:CAP:HARD:PORT:COUN?')   # number of ports
    """

    default_config: dict[str, Any] = {
        "sweep_type": "LIN",
        "measurements": (("default trace", "S11"),),
        "power": -40.0,
        "if_bandwidth": 5000,
        "freq_spec": (10e8, 20e9, 2001),
        "average": 1,
    }

    def __init__(
        self,
        name: str,
        address: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            address,
            min_freq=9e3,      # VERIFY
            max_freq=20e9,     # VERIFY
            min_power=-100,    # VERIFY
            max_power=13,      # VERIFY
            nports=4,          # VERIFY - P502xA is the 4-port line
            **kwargs,
        )
        self._visa_address = address
        self._traces_by_label: dict[str, KeysightPNATrace] = {}

        # SOUR:POW<port>:SLOP[:STAT] - power slope compensation (e.g. for
        # cable loss), single implicit channel like the rest of this
        # driver, hardcoded to port 1 like every call site in
        # measurement_scripts/ (`AgilentPNAChannel.port` also defaults to 1
        # and nothing overrides it there either).
        self.power_slope: Parameter = self.add_parameter(
            "power_slope",
            label="Power slope",
            unit="dB/GHz",
            get_cmd="SOUR:POW1:SLOP?",
            get_parser=float,
            set_cmd="SOUR:POW1:SLOP {}",
        )
        """Parameter power_slope"""

        self.power_slope_state: Parameter = self.add_parameter(
            "power_slope_state",
            label="Power slope enabled",
            get_cmd="SOUR:POW1:SLOP:STAT?",
            get_parser=int,
            set_cmd="SOUR:POW1:SLOP:STAT {}",
        )
        """Parameter power_slope_state"""

        self._config = dict(self.default_config)
        if config:
            for key in self.default_config:
                if key in config:
                    self._config[key] = config[key]

        self.reset_config()
        self.configure(self._config)

    def get_config_info(self) -> dict[str, Any]:
        """Same shape as `instruments_old.basic_instrument.BasicInstrument
        .get_config_info`, so this instrument can be dropped straight into
        `BaseMeasurement.instruments`."""
        return {
            "visa_address": self._visa_address,
            "nickname": self.name,
            "config": self._config,
        }

    def reset_config(self) -> None:
        """Presets the VNA (leaving its one factory-default trace in
        place - deleting *every* trace here would leave `configure_
        measurements()`'s `add_trace()` calls with no baseline trace to
        diff against, which the native driver's `add_trace()` can't
        recover from), then turns output on, widens data precision to
        REAL,64 (needed for accurate frequency data - see
        `instruments_old/exopy_hqc_legacy/drivers/visa/agilent_pna.py
        ::AgilentPNA.data_format`), and disables `auto_sweep` so reading
        out several named traces after one `run_averaging()` doesn't
        silently re-trigger a full sweep per trace (see `read_raw_data`).
        Matches `VNA_P5024A.reset_config`."""
        self.write("SYST:PRESet")
        time.sleep(1.0)  # let the PNA settle before further SCPI - VERIFY
        self._traces_by_label = {}
        self.output(True)
        self.write("FORM REAL,64")
        self.auto_sweep(False)

    def configure(self, config: dict[str, Any]) -> None:
        """Apply a config dict. See `default_config` / class docstring for
        keys. Matches `VNA_P5024A.set_config`."""
        self._config.update(config)
        if "sweep_type" in config:
            self.sweep_type(config["sweep_type"])
        if "freq_spec" in config:
            sweep_type = self._config.get("sweep_type", "LIN")
            if sweep_type == "LIN":
                freq_start, freq_stop, n_freqs = config["freq_spec"]
                self.start(freq_start)
                self.stop(freq_stop)
                self.points(n_freqs)
            elif sweep_type == "CW":
                freq_cw, n_pts = config["freq_spec"]
                self.cw(freq_cw)
                self.points(n_pts)
            else:
                raise ValueError(f"Unknown sweep_type: {sweep_type!r}")
        if "measurements" in config:
            self.configure_measurements(config["measurements"])
        if "power" in config:
            self.power(config["power"])
        if "if_bandwidth" in config:
            self.if_bandwidth(config["if_bandwidth"])
        if "average" in config:
            n_avg = config["average"]
            if n_avg == 1:
                self.averages_enabled(False)
            else:
                self.averages_enabled(True)
                self.averages(n_avg)

    def configure_measurements(
        self, measurements: tuple[tuple[str, str], ...]
    ) -> None:
        """Create one named trace per `(label, S-parameter)` pair, e.g.
        `(('Sig1Sig1', 'S23'), ('Sig2Sig1', 'S43'), ...)`, all bound into
        window 1 with a shared Y scale. Replaces
        `AgilentPNAChannel.create_meas`/`bind_meas_to_window`/
        `couple_scale`; `label -> trace` is kept on `self._traces_by_label`
        for `read_raw_data`.

        Reuses whatever traces already exist on the instrument (at least
        one, from `reset_config()`'s preset) for the first N labels before
        calling `add_trace()` for the rest, and deletes any left over -
        rather than deleting everything first and adding N fresh ones,
        which would momentarily leave the instrument with zero traces
        (a state the native driver's own `add_trace()`/`traces` machinery
        can't recover from)."""
        self._traces_by_label = {}
        self.write("DISPlay:WINDow1 ON")
        available = list(self.traces)
        for trace_num, (label, sparam) in enumerate(measurements, start=1):
            trace = available.pop(0) if available else self.add_trace()
            trace.trace(sparam)
            self._traces_by_label[label] = trace
            self.write(
                f"DISPlay:WINDow1:TRACe{trace_num}:FEED '{trace.trace_name}'"
            )
        for leftover in available:
            self.write(f"CALCulate:PARameter:DELete '{leftover.trace_name}'")
        self.write("DISPlay:WINDow1:TRACe:y:COUP:METH WIND")

    def read_raw_data(self, label: str) -> np.ndarray:
        """Complex raw I/Q data for one named measurement configured via
        `configure_measurements`. Matches
        `AgilentPNAChannel.read_raw_data`."""
        return self._traces_by_label[label].polar()

    def read_freq_data(self) -> np.ndarray:
        """Frequency axis for the current sweep. Matches
        `AgilentPNAChannel.read_freq_data`."""
        return self.frequency_axis()

    def run_averaging(self) -> None:
        """Restart averaging and block until the group of sweeps completes
        - covers every trace on the (single, implicit) channel in one
        trigger. Matches `AgilentPNAChannel.run_averaging`, delegating to
        the native driver's `KeysightPNATrace.run_sweep()` which already
        does the same thing (reset averages, set group-trigger-count to
        the average count, poll `sweep_mode` until HOLD).

        NOTE: deliberately does *not* use the `self.run_sweep` shortcut the
        base class wires up at connect time - that's bound to whatever
        trace happened to exist before `reset_config()`/
        `configure_measurements()` ran, which has since been deleted."""
        if not self._traces_by_label:
            raise RuntimeError(
                "No measurements configured - call configure_measurements() first."
            )
        next(iter(self._traces_by_label.values())).run_sweep()

    def safe_shutdown(self) -> None:
        """Called by the measurement harness's `safe_run()` on *every*
        exit from a run - a clean finish as much as an error/abort, not
        error/abort only. Turns RF output off; lighter than
        `reset_config()`, which would also wipe the configured traces.
        """
        self.output(False)
