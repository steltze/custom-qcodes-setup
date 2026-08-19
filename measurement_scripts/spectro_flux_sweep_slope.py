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

        super().__init__(save_path, circuit_path)

    def execute(self):
        self._setup_sweep_coordinates()
        self._setup_instruments()
        self._setup_qcodes_experiment()
        self._run_sweep()

    def _setup_sweep_coordinates(self):
        """Every array that defines this sweep. Pure math - no
        instrument, qcodes, or h5py calls."""
        self._dc_current = np.linspace(
            self.params["current_start"],
            self.params["current_end"],
            self.params["n_current"],
        )
        self._n_current = self.params["n_current"]
        self._width = len(str(self._n_current))

    def _setup_instruments(self):
        """Configure the VNA's power slope and take the one reference
        sweep whose frequency axis is shared by the whole measurement.
        Only talks to `self.vna`."""
        self.vna.power_slope(self.params["power_slope"])
        self.vna.power_slope_state(1)
        self._freq_data = self.vna.read_freq_data()

    def _setup_qcodes_experiment(self):
        """Open the sqlite .db/experiment this run gets added to, and
        snapshot the station + this sweep's params as run metadata. Only
        talks to the qcodes API."""
        self._experiment = open_experiment(
            self.save_file_path,
            experiment_name="spectro_flux_sweep_slope",
            sample_name=Path(self.circuit_path).stem,
        )
        self._station = Station(self.vna, self.yoko)
        self._run_metadata = instrument_metadata(
            self.instruments, self.params, self.circuit_path
        )

    def _setup_h5(self, file):
        """Write the sweep's one static axis into the already-open file.
        Only talks to h5py."""
        data_group = file["data"]
        data_group.create_dataset("vna frequencies", data=self._freq_data)

    def _run_sweep(self):
        """Measure once per current, then hand the result to both the
        qcodes run and the legacy .h5. `PeriodicFlush` only throttles how often
        that's pushed to disk with an OS-level flush, it never holds
        extra points in memory."""

        with safe_run(self.instruments), h5.File(self.save_file_path, "r+") as file:
            self._setup_h5(file)
            data_group = file["data"]
            meas = self._setup_qcodes_run()
            flush_gate = PeriodicFlush(interval=self.params.get("flush_interval", 300.0))

            with meas.run() as datasaver:
                for key, value in self._run_metadata.items():
                    datasaver.dataset.add_metadata(key, value)

                for loop_number, current in enumerate(tqdm(self._dc_current), start=1):
                    # take measurement
                    trace_data, time_str = self._measure_one_point(current)

                    # old .h5
                    self._save_point_to_h5(
                        data_group, loop_number, current, time_str, trace_data
                    )
                    # qcodes .db
                    self._save_point_to_db(datasaver, current, time_str, trace_data)

                    if flush_gate.due():
                        file.flush()
                        datasaver.flush_data_to_database(block=False)

            file.flush()

    def _measure_one_point(self, current):
        """Set the DC current and read back the VNA trace for every
        configured measurement. Only talks to `self.yoko`/`self.vna` - no
        qcodes, no h5py."""
        self.yoko.current(current)
        self.vna.run_averaging()
        time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        trace_data = {
            label: self.vna.read_raw_data(label) for label in self._vna_measurement_labels
        }
        return trace_data, time_str

    def _setup_qcodes_run(self):
        """Register the Measurement + Parameters for this sweep. Sets
        `self._qc_*`, read by `_save_point_to_db`."""
        vna_meas = self._vna_measurement_labels
        self._qc_current_param = Parameter(
            "current", label="DC current", unit="A", get_cmd=None, set_cmd=None
        )
        self._qc_freq_param = Parameter(
            "frequency", label="VNA frequency", unit="Hz", get_cmd=None, set_cmd=None
        )
        self._qc_time_param = Parameter(
            "time", label="Timestamp", get_cmd=None, set_cmd=None
        )
        self._qc_trace_params = {
            label: Parameter(label, label=label, unit="", get_cmd=None, set_cmd=None)
            for label in vna_meas
        }

        meas = Measurement(
            name="spectro_flux_sweep_slope", exp=self._experiment, station=self._station
        )
        meas.register_parameter(self._qc_current_param, paramtype="numeric")
        meas.register_parameter(self._qc_freq_param, paramtype="array")
        meas.register_parameter(
            self._qc_time_param, setpoints=(self._qc_current_param,), paramtype="text"
        )
        for param in self._qc_trace_params.values():
            meas.register_parameter(
                param, setpoints=(self._qc_current_param, self._qc_freq_param), paramtype="array"
            )
        return meas

    def _save_point_to_db(self, datasaver, current, time_str, trace_data):
        """One point -> one row in the qcodes run."""
        results = [
            (self._qc_current_param, current),
            (self._qc_freq_param, self._freq_data),
            (self._qc_time_param, time_str),
        ]
        results += [
            (self._qc_trace_params[label], trace_data[label])
            for label in self._vna_measurement_labels
        ]
        datasaver.add_result(*results)

    def _save_point_to_h5(self, data_group, loop_number, current, time_str, trace_data):
        """One point -> one named dataset."""
        vna_meas = self._vna_measurement_labels
        data = np.stack([trace_data[label] for label in vna_meas], axis=-1)
        dset = data_group.create_dataset(str(loop_number).zfill(self._width), data=data)
        dset.attrs["time"] = time_str

        dset.attrs["Current (mA)"] = float(current)
        for i_meas, label in enumerate(vna_meas):
            dset.attrs[f"Data col {i_meas}"] = label
