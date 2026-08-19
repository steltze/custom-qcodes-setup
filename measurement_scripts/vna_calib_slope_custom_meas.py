import sys
from copy import deepcopy
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
from qcodes_utils.measurement_run import (  # noqa: E402
    instrument_metadata,
    open_experiment,
    safe_run,
)

from .base_measurement import BaseMeasurement

_MATRIX_POSITION = {
    "Sig1Sig1": (0, 0),
    "Sig1Sig2": (0, 1),
    "Sig2Sig1": (1, 0),
    "Sig2Sig2": (1, 1),
}


class VNACalibSlopeCustomMeas(BaseMeasurement):
    def __init__(self, save_path, circuit_path, vna_params, sweep_params):
        """
        vna_params:dict('visa_address':str, 'nickname':str, config:dict)
        'sweep_params':dict(
                'pts_list': list(int),
                'bw_list': list(int),
                'power_slope':float,
                'measurements': for example (
            ('Sig1Sig1', 'S31'),
            ('Sig2Sig1', 'S41'),
            ('Sig1Sig2', 'S32'),
            ('Sig2Sig2', 'S42')
        )
                )
            )

        Each `(pts, bw)` pair is captured as its own QCoDeS run (sqlite
        `.db`, the source of truth) and immediately exported to its own
        legacy-shaped `.h5` file, `<save_path>_<pts>pts.h5` - same
        one-file-per-iteration layout the old script wrote. `save_path`
        itself only ever receives the `BaseMeasurement` metadata block
        (its `data` group stays empty), matching the old script exactly.
        """
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        self.vna = KeysightP5024A(
            name=vna_params_mod["nickname"],
            address=vna_params_mod["visa_address"],
            config=vna_params_mod.get("config"),
        )
        self.instruments = (self.vna,)
        super().__init__(save_path, circuit_path)

    def execute(self):
        self._setup_instruments()
        self._setup_qcodes_experiment()
        self._run_sweep()

    def _setup_instruments(self):
        """Configure the VNA's power slope. Only talks to `self.vna`."""
        self.vna.power_slope_state(1)
        self.vna.power_slope(self.params["power_slope"])

    def _setup_qcodes_experiment(self):
        """Open the sqlite .db/experiment new runs get added to, and
        snapshot the station + this sweep's params as run metadata. Only
        talks to the qcodes API."""
        self._experiment = open_experiment(
            self.save_file_path,
            experiment_name="vna_calib_slope_custom_meas",
            sample_name=Path(self.circuit_path).stem,
        )
        self._station = Station(self.vna)
        self._run_metadata = instrument_metadata(
            self.instruments, self.params, self.circuit_path
        )

    def _run_sweep(self):
        """Measure once per `(pts, bw)` pair, then hand the result to
        both the legacy .h5 and the qcodes run. Two separate calls per
        pair (not interleaved point-by-point like the other 3 scripts):
        each `(pts, bw)` acquisition is one atomic VNA sweep with no
        useful mid-point to checkpoint, so there's nothing a db
        round-trip or finer interleaving would add here."""
        n_pts_list = self.params["pts_list"]
        bw_list = self.params["bw_list"]
        with safe_run(self.instruments):
            bar = tqdm()
            bar.reset(total=len(n_pts_list))
            for n_pt, bw in zip(n_pts_list, bw_list):

                trace_data, freq_data = self._measure_one_sweep(n_pt, bw)

                self._save_to_h5(n_pt, freq_data, trace_data)

                self._save_to_db(n_pt, bw, freq_data, trace_data)

                bar.update()
            bar.refresh()

    def _measure_one_sweep(self, n_pt, bw):
        """Set the sweep's point count/bandwidth and read back the VNA
        trace for every configured measurement. Only talks to
        `self.vna` - no qcodes, no h5py."""
        self.vna.if_bandwidth(bw)
        self.vna.points(n_pt)
        self.vna.run_averaging()
        freq_data = self.vna.read_freq_data()
        trace_data = {
            label: self.vna.read_raw_data(label) for label in _MATRIX_POSITION
        }
        return trace_data, freq_data

    def _save_to_db(self, n_pt, bw, freq_data, trace_data):
        """One full run (this whole acquisition is a single row) into
        the sqlite .db."""
        freq_param = Parameter(
            "frequency", label="VNA frequency", unit="Hz", get_cmd=None, set_cmd=None
        )
        trace_params = {
            label: Parameter(label, label=label, unit="", get_cmd=None, set_cmd=None)
            for label in trace_data
        }

        meas = Measurement(
            name="vna_calib_slope_custom_meas", exp=self._experiment, station=self._station
        )
        meas.register_parameter(freq_param, paramtype="array")
        for param in trace_params.values():
            meas.register_parameter(param, setpoints=(freq_param,), paramtype="array")

        with meas.run() as datasaver:
            results = [(freq_param, freq_data)]
            results += [(trace_params[label], trace_data[label]) for label in trace_data]
            datasaver.add_result(*results)
            for key, value in self._run_metadata.items():
                datasaver.dataset.add_metadata(key, value)
            datasaver.dataset.add_metadata("n_pts", str(n_pt))
            datasaver.dataset.add_metadata("if_bandwidth", str(bw))

    def _save_to_h5(self, n_pt, freq_data, trace_data):
        """Written straight from the values just measured (the same ones
        passed to `datasaver.add_result` in `_save_to_db`) rather than
        reading them back from the db, since there's nothing a db
        round-trip would add here (unlike the two whole-sweep-is-one-run
        scripts, where writing point-by-point during the loop is what
        protects against a mid-sweep crash)."""
        data = np.empty((len(freq_data), 2, 2), dtype="complex128")
        for label, (i, j) in _MATRIX_POSITION.items():
            data[:, i, j] = trace_data[label]

        file_path = self.save_file_path[:-3] + f"_{n_pt}pts.h5"
        with h5.File(file_path, "w") as file:
            data_group = file.create_group("data")
            data_group.create_dataset("Vna frequencies (Hz)", data=freq_data)
            dset = data_group.create_dataset("data", data=data, dtype="complex128")
            dset.dims[0].label = "Vna frequencies (Hz)"
            dset.dims[1].label = "X in SigXSigY"
            dset.dims[2].label = "Y in SigXSigY"
