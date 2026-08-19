import json
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

from instruments.KeysightM8195A import KeysightM8195A  # noqa: E402
from instruments.KeysightP5024A import KeysightP5024A  # noqa: E402
from instruments.YokogawaGS200 import YokogawaGS200  # noqa: E402
from qcodes_utils.measurement_run import (  # noqa: E402
    PeriodicFlush,
    instrument_metadata,
    open_experiment,
    safe_run,
)

from .base_measurement import BaseMeasurement

#: order the old script assembled the 2x2 crosstalk matrix in, before
#: transposing to (main_amp, freq, 2, 2) - see _save_point_to_h5 below.
_MATRIX_POSITION = {
    "Sig1Sig1": (0, 0),
    "Sig1Sig2": (0, 1),
    "Sig2Sig1": (1, 0),
    "Sig2Sig2": (1, 1),
}

#: AWG channel amplitude hardware floor (Vpk-pk). See class docstring.
_MIN_AMP = 75e-3


class SpectroAWGPumpSweepFIRSimpNOCompSweepFlux(BaseMeasurement):
    def __init__(self, save_path, circuit_path, vna_params, awg_params, dc_params, sweep_params):
        """
        vna_params:dict('visa_address':str, 'nickname':str, config:dict)
            'measurements' will be overwritten.
        awg_params:dict(
            'visa_address':str,
            'nickname':str,
            'config':dict
            )
        'sweep_params':dict(
                'main_channel':str,
                'freqs': list(float) of len n_freq (def of n_freq)
                'main_amp_starts':list(float) of len n_freq
                'main_amp_ends':list(float) of len n_freq,
                'n_main_amp':int,
                'power_slope':float
                )
            )

        NOTE on sub-75mV pump amplitudes: the old script tried to
        compensate amplitudes below the AWG's 75mV hardware floor via a
        `fir_scale` config key - that key doesn't exist anywhere in the
        currently installed pyarbtools/AWG_M8195A driver (`configure()`
        raises KeyError on it), so that path was already broken. Per
        explicit instruction, this rewrite simply *skips* any requested
        amplitude below 75mV (logged into each run's
        'skipped_main_amps_below_75mV' metadata, and into
        `data.attrs['skipped_main_amp_indices']` in the exported `.h5`)
        rather than silently measuring it at the wrong, floored amplitude.
        """
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        vna_config = dict(vna_params_mod.get("config") or {})
        vna_config["measurements"] = (
            ("Sig1Sig1", "S23"),
            ("Sig2Sig1", "S43"),
            ("Sig1Sig2", "S21"),
            ("Sig2Sig2", "S41"),
        )
        vna_params_mod["config"] = vna_config
        self._vna_measurement_labels = tuple(label for label, _ in vna_config["measurements"])

        self.vna = KeysightP5024A(
            name=vna_params_mod["nickname"],
            address=vna_params_mod["visa_address"],
            config=vna_config,
        )
        self.pump = KeysightM8195A(
            name=awg_params["nickname"],
            address=awg_params["visa_address"],
            config=awg_params.get("config"),
        )
        self.yoko = YokogawaGS200(
            name=dc_params["nickname"],
            address=dc_params["visa_address"],
            config=dc_params.get("config"),
        )
        self.instruments = (self.vna, self.pump, self.yoko)
        self.main_ch = self.params["main_channel"]
        
        super().__init__(save_path, circuit_path)

    def execute(self):
        self._setup_sweep_coordinates()
        self._setup_instruments()
        self._setup_qcodes_experiment()
        self._setup_h5_axes()
        self._run_sweep()

    def _setup_sweep_coordinates(self):
        """Every array that defines this sweep. Pure math - no
        instrument, qcodes, or h5py calls."""
        self._dc_current = np.linspace(
            self.params["current_start"], self.params["current_end"], self.params["n_current"]
        )
        self._pump_freqs = np.asarray(self.params["freqs"])
        self._n_freq = len(self._pump_freqs)
        self._n_main_amp = self.params["n_main_amp"]

        self._main_pump_amps = np.zeros((self._n_freq, self._n_main_amp))
        for i in range(self._n_freq):
            self._main_pump_amps[i] = np.linspace(
                self.params["main_amp_starts"][i],
                self.params["main_amp_ends"][i],
                self._n_main_amp,
            )

    def _setup_instruments(self):
        """Configure the VNA's power slope, take the one reference sweep
        whose frequency axis is shared by the whole measurement, and put
        the AWG at its safe starting amplitude. Only talks to
        `self.vna`/`self.pump`."""
        self.vna.power_slope(self.params["power_slope"])
        self.vna.power_slope_state(1)
        self.vna.run_averaging()
        self._freq_data = self.vna.read_freq_data()

        self._amp_param = getattr(self.pump, f"amplitude_{self.main_ch}")
        self._amp_param(_MIN_AMP)

    def _setup_qcodes_experiment(self):
        """Open the sqlite .db/experiment new runs get added to, and
        snapshot the station + this sweep's params as run metadata. Only
        talks to the qcodes API."""
        self._experiment = open_experiment(
            self.save_file_path,
            experiment_name="spectro_awgPump_sweep_flux",
            sample_name=Path(self.circuit_path).stem,
        )
        self._station = Station(self.vna, self.pump, self.yoko)
        self._run_metadata = instrument_metadata(
            self.instruments, self.params, self.circuit_path
        )

    def _setup_h5_axes(self):
        """Write the sweep's static axes into `self.save_file_path`'s
        `data` group - the four datasets the old script wrote once,
        before ever touching an individual `(current, freq)` file. Only
        talks to h5py."""
        with h5.File(self.save_file_path, "r+") as file:
            data_group = file["data"]
            data_group.create_dataset("Vna frequencies (Hz)", data=self._freq_data)
            data_group.create_dataset("Pump frequencies (Hz)", data=self._pump_freqs)
            data_group.create_dataset(
                "Main pump amps (Vpk-pk)", data=self._main_pump_amps
            )
            data_group.create_dataset("DC current (A)", data=self._dc_current)

    def _run_sweep(self):
        """Drive the DC current and pump frequency/segment, delegating
        each `(current, freq)` pair's amplitude sweep to
        `_run_one_file`."""
        seg_id_main = None
        total_loops = len(self._dc_current) * self._n_freq * self._n_main_amp

        with safe_run(self.instruments):
            bar = tqdm()
            bar.reset(total=total_loops)
            for id_current, current in enumerate(self._dc_current):
                self.yoko.current(current)
                for id_freq, freq in enumerate(self._pump_freqs):
                    if seg_id_main is not None:
                        self.pump.delete_segment(seg_id_main, ch=self.main_ch)
                    seg_id_main = self.pump.send_sine(freq, 0, self.main_ch, _MIN_AMP)

                    file_path = (
                        self.save_file_path[:-3] + f"_current{id_current}_freq{id_freq}.h5"
                    )
                    self._run_one_file(seg_id_main, self._main_pump_amps[id_freq], bar, file_path)
            bar.refresh()
            self.pump.clear_all_wfm()

    def _run_one_file(self, seg_id_main, requested_amps, bar, file_path):
        """Measure one `(current, freq)` pair's amplitude sweep, writing
        into its own qcodes run *and* its own legacy-shaped `.h5` file
        point by point as each amplitude is measured - not rebuilt/
        re-read from the db - so a crash mid-sub-run only loses that
        one file's still-unmeasured amplitudes, not the ones already on
        disk. Amplitudes skipped for being below the 75mV floor are left
        as zero rows, flagged in `data.attrs['skipped_main_amp_indices']`
        rather than silently indistinguishable from a real all-zero
        measurement."""
        n_main_amp = len(requested_amps)
        meas = self._setup_qcodes_run()
        flush_gate = PeriodicFlush(interval=self.params.get("flush_interval", 10.0))
        skipped_values: list[float] = []
        skipped_indices: list[int] = []

        with h5.File(file_path, "w") as file:
            dset = self._setup_h5_dataset(file, n_main_amp)

            with meas.run() as datasaver:
                for key, value in self._run_metadata.items():
                    datasaver.dataset.add_metadata(key, value)

                for idx, main_amp in enumerate(requested_amps):
                    if main_amp < _MIN_AMP:
                        skipped_values.append(float(main_amp))
                        skipped_indices.append(idx)
                        bar.update()
                        continue

                    # take measurement
                    trace_data = self._measure_one_point(main_amp, seg_id_main)

                    time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    # old .h5
                    self._save_point_to_h5(dset, idx, time_str, trace_data)
                    # qcodes .db
                    self._save_point_to_db(datasaver, main_amp, time_str, trace_data)

                    bar.update()
                    if flush_gate.due():
                        file.flush()
                        datasaver.flush_data_to_database(block=False)

                datasaver.dataset.add_metadata(
                    "skipped_main_amps_below_75mV", json.dumps(skipped_values)
                )

            dset.attrs["skipped_main_amp_indices"] = np.array(skipped_indices, dtype=int)
            file.flush()

    def _measure_one_point(self, main_amp, seg_id_main):
        """Set the pump amplitude, play it, and read back the VNA trace
        for every configured measurement. Only talks to
        `self.pump`/`self.vna` - no qcodes, no h5py."""
        self._amp_param(main_amp)
        self.pump.play(seg_id_main, self.main_ch)
        self.pump.ask_if_done()
        
        self.vna.run_averaging()
        trace_data = {
            label: self.vna.read_raw_data(label) for label in self._vna_measurement_labels
        }
        return trace_data

    def _setup_qcodes_run(self):
        """Register a fresh Measurement + Parameters for one `(current,
        freq)` sub-run. Sets `self._qc_*`, read by `_save_point_to_db`."""
        vna_labels = self._vna_measurement_labels
        self._qc_main_amp_param = Parameter(
            "main_amp", label="Pump amplitude", unit="Vpk-pk", get_cmd=None, set_cmd=None
        )
        self._qc_freq_param = Parameter(
            "vna_frequency", label="VNA frequency", unit="Hz", get_cmd=None, set_cmd=None
        )
        self._qc_time_param = Parameter("time", label="Timestamp", get_cmd=None, set_cmd=None)
        self._qc_trace_params = {
            label: Parameter(label, label=label, unit="", get_cmd=None, set_cmd=None)
            for label in vna_labels
        }

        meas = Measurement(
            name="spectro_awgPump_sweep_flux", exp=self._experiment, station=self._station
        )
        meas.register_parameter(self._qc_main_amp_param, paramtype="numeric")
        meas.register_parameter(self._qc_freq_param, paramtype="array")
        meas.register_parameter(
            self._qc_time_param, setpoints=(self._qc_main_amp_param,), paramtype="text"
        )
        for param in self._qc_trace_params.values():
            meas.register_parameter(
                param,
                setpoints=(self._qc_main_amp_param, self._qc_freq_param),
                paramtype="array",
            )
        return meas

    def _save_point_to_db(self, datasaver, main_amp, time_str, trace_data):
        """One point -> one row in the qcodes run."""
        datasaver.add_result(
            (self._qc_main_amp_param, main_amp),
            (self._qc_freq_param, self._freq_data),
            (self._qc_time_param, time_str),
            *[
                (self._qc_trace_params[label], trace_data[label])
                for label in self._vna_measurement_labels
            ],
        )

    def _setup_h5_dataset(self, file, n_main_amp):
        """Pre-allocate one sub-run's `data` array at its full final
        shape, exactly like the old script did."""
        n_freq = len(self._freq_data)
        dset = file.create_dataset("data", (n_main_amp, n_freq, 2, 2), dtype="complex128")
        dset.dims[0].label = "Main pump amps"
        dset.dims[1].label = "Vna frequencies (Hz)"
        dset.dims[2].label = "X in SigXSigY"
        dset.dims[3].label = "Y in SigXSigY"
        return dset

    def _save_point_to_h5(self, dset, idx, time_str, trace_data):
        """One point -> one cell of the pre-allocated `data` array."""
        n_freq = len(self._freq_data)
        block = np.empty((n_freq, 2, 2), dtype="complex128")
        for label, (i, j) in _MATRIX_POSITION.items():
            block[:, i, j] = trace_data[label]
        dset[idx] = block
        dset.attrs["time"] = time_str
