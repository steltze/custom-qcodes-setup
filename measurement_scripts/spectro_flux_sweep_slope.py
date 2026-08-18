import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import h5py as h5
import numpy as np
from tqdm.notebook import tqdm

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qcodes.dataset import Measurement
from qcodes.parameters import Parameter
from qcodes.station import Station

from instruments.KeysightP5024A import KeysightP5024A  # noqa: E402
from instruments.YokogawaGS200 import YokogawaGS200  # noqa: E402
from qcodes_utils.measurement_run import (  # noqa: E402
    PeriodicFlush,
    instrument_metadata,
    open_experiment,
    safe_run,
)

from .base_measurement import BaseMeasurement


class SpectroDCSweepSlope(BaseMeasurement):
    def __init__(self, save_path, circuit_path, vna_params, dc_params, sweep_params):
        """
        vna_params:dict('visa_address':str, 'nickname':str, config:dict)
            'measurements' will be overwritten.
        dc_params:dict(
            'visa_address':str,
            'nickname':str,
            'config':dict
            )
        'sweep_params':dict(
                'channel':int,
                'current_start':float,
                'current_end':float,
                'n_current':int,
                'power_slope':float)
            )
        """
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        vna_config = dict(vna_params_mod.get("config") or {})
        # Antho's device mapping:
        if vna_config.get("line") == "pump":
            vna_config["measurements"] = (
                ("Sig1Sig1", "S43"),
                ("Sig1Sig2", "S41"),
            )
        else:
            vna_config["measurements"] = (
                ("Sig1Sig1", "S41"),
                ("Sig2Sig1", "S21"),
                ("Sig1Sig2", "S43"),
                ("Sig2Sig2", "S23"),
            )
        vna_params_mod["config"] = vna_config
        self._vna_measurement_labels = tuple(
            label for label, _ in vna_config["measurements"]
        )

        self.vna = KeysightP5024A(
            name=vna_params_mod["nickname"],
            address=vna_params_mod["visa_address"],
            config=vna_config,
        )
        self.yoko = YokogawaGS200(
            name=dc_params["nickname"],
            address=dc_params["visa_address"],
            config=dc_params.get("config"),
        )
        self.instruments = (self.vna, self.yoko)
        # call the super at the end!
        super().__init__(save_path, circuit_path)

    def execute(self):
        self.vna.power_slope(self.params["power_slope"])
        self.vna.power_slope_state(1)

        dc_current = np.linspace(
            self.params["current_start"],
            self.params["current_end"],
            self.params["n_current"],
        )
        vna_meas = self._vna_measurement_labels
        freq_data = self.vna.read_freq_data()

        experiment = open_experiment(
            self.save_file_path,
            experiment_name="spectro_flux_sweep_slope",
            sample_name=Path(self.circuit_path).stem,
        )
        station = Station(self.vna, self.yoko)
        run_metadata = instrument_metadata(
            self.instruments, self.params, self.circuit_path
        )

        current_param = Parameter(
            "current", label="DC current", unit="A", get_cmd=None, set_cmd=None
        )
        freq_param = Parameter(
            "frequency", label="VNA frequency", unit="Hz", get_cmd=None, set_cmd=None
        )
        time_param = Parameter(
            "time", label="Timestamp", get_cmd=None, set_cmd=None
        )
        trace_params = {
            label: Parameter(label, label=label, unit="", get_cmd=None, set_cmd=None)
            for label in vna_meas
        }

        meas = Measurement(
            name="spectro_flux_sweep_slope", exp=experiment, station=station
        )
        meas.register_parameter(current_param, paramtype="numeric")
        meas.register_parameter(freq_param, paramtype="array")
        meas.register_parameter(time_param, setpoints=(current_param,), paramtype="text")
        for param in trace_params.values():
            meas.register_parameter(
                param, setpoints=(current_param, freq_param), paramtype="array"
            )

        # NOTE: the old script called `self.yoko.reset_config` *before* the
        # sweep too - but with no `()`, so it was a no-op (referenced the
        # bound method, never called it). Actually calling it here would
        # turn the DC source's output off right before sweeping current
        # through it, which can't have been the intent - the source was
        # left exactly as `__init__`/`configure()` set it up (output on,
        # per `dc_params['config']['output']`), which is what happened in
        # practice. Only the *end-of-run* safe-off (below, and via
        # `safe_run` on any abort) is actually applied here.

        # This sweep can run for hours (the flux-map notebook this replaces
        # ran 5.5h for 501 points) with no natural checkpoint, so the
        # legacy `.h5` is written point by point into the *same* file
        # `BaseMeasurement` already created - not rebuilt/re-read from the
        # db - exactly like the old script did, so a crash/power-loss
        # loses at most the last few points, never the whole file.
        # `PeriodicFlush` only throttles how often that's pushed to disk
        # with an OS-level flush, it never holds extra points in memory.
        n_current = self.params["n_current"]
        width = len(str(n_current))
        flush_gate = PeriodicFlush(interval=self.params.get("flush_interval", 15.0))

        with safe_run(self.instruments), h5.File(self.save_file_path, "r+") as file:
            data_group = file["data"]
            data_group.create_dataset("vna frequencies", data=freq_data)

            with meas.run() as datasaver:
                for key, value in run_metadata.items():
                    datasaver.dataset.add_metadata(key, value)

                for loop_number, current in enumerate(tqdm(dc_current), start=1):
                    self.yoko.current(current)
                    self.vna.run_averaging()
                    time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    trace_data = {
                        label: self.vna.read_raw_data(label) for label in vna_meas
                    }

                    results = [
                        (current_param, current),
                        (freq_param, freq_data),
                        (time_param, time_str),
                    ]
                    results += [
                        (trace_params[label], trace_data[label]) for label in vna_meas
                    ]
                    datasaver.add_result(*results)

                    data = np.stack([trace_data[label] for label in vna_meas], axis=-1)
                    dset = data_group.create_dataset(
                        str(loop_number).zfill(width), data=data
                    )
                    dset.attrs["time"] = time_str
                    # Reproduces the old script's own attribute exactly,
                    # including what looks like a units bug (current here
                    # is already in amps, so this is amps*1e-3, not really
                    # an amp->mA conversion) - flagged, not silently
                    # changed.
                    dset.attrs["Current (mA)"] = float(current) * 1e-3
                    for i_meas, label in enumerate(vna_meas):
                        dset.attrs[f"Data col {i_meas}"] = label

                    if flush_gate.due():
                        file.flush()
                        datasaver.flush_data_to_database(block=False)

            file.flush()
            self.yoko.reset_config()
