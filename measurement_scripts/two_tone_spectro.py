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

from instruments.AnaPicoAPUASYN20 import AnaPicoAPUASYN20  # noqa: E402
from instruments.KeysightP5024A import KeysightP5024A  # noqa: E402
from instruments.YokogawaGS200 import YokogawaGS200  # noqa: E402
from qcodes_utils.measurement_run import (  # noqa: E402
    PeriodicFlush,
    instrument_metadata,
    open_experiment,
    safe_run,
)

from .base_measurement import BaseMeasurement


class TwoToneSpectro(BaseMeasurement):
    def __init__(
        self, save_path, circuit_path, vna_params, pico_params, dc_params, sweep_params
    ):
        """
        vna_params:dict(
            'visa_address':str,
            'nickname':str,
            config:dict)
        pico_params:dict(
            'visa_address':str,
            'nickname':str,
            'config':dict
            )
        dc_params:dict(
        'visa_address':str,
        'nickname':str,
        'config':dict
        )
        sweep_params:dict(
            'pico_freqs': shape (nfreqs,)
            'currents': shape (ncurrents,)
            'vna_freqs': shape (ncurrents,)
            'pico_powers': shape (nfreqs,)
        )
        """
        self.params = sweep_params
        vna_params_mod = deepcopy(vna_params)
        vna_config = dict(vna_params_mod["config"])
        vna_config["sweep_type"] = "CW"
        vna_params_mod["config"] = vna_config
        self._vna_measurement_labels = tuple(
            label for label, _ in vna_config["measurements"]
        )
        if len(self._vna_measurement_labels) != 1:
            # The old script's `get_vna_data()` stacked every configured
            # measurement, but the fixed-shape `dset[i, j] = data`
            # assignment it wrote into only actually worked (via numpy
            # broadcasting) for exactly one - it silently only ever
            # supported a single measurement. Surface that rather than
            # reproduce it as silent data corruption.
            raise NotImplementedError(
                "TwoToneSpectro only supports a single vna measurement "
                f"(got {self._vna_measurement_labels!r}) - this matches the "
                "old script's actual (if undocumented) limitation."
            )

        # replaces pico_params['config'] outright, same as the old script
        pico_config = {
            "frequencies": (self.params["pico_freqs"][0],),
            "powers": (self.params["pico_powers"][0],),
        }

        self.vna = KeysightP5024A(
            name=vna_params_mod["nickname"],
            address=vna_params_mod["visa_address"],
            config=vna_config,
        )
        self.pico = AnaPicoAPUASYN20(
            name=pico_params["nickname"],
            address=pico_params["visa_address"],
            config=pico_config,
        )
        self.yoko = YokogawaGS200(
            name=dc_params["nickname"],
            address=dc_params["visa_address"],
            config=dc_params.get("config"),
        )
        self.instruments = (self.vna, self.pico, self.yoko)
        # call the super at the end!
        super().__init__(save_path, circuit_path)

    def execute(self):
        currents = np.array(self.params["currents"])
        vna_freqs = np.array(self.params["vna_freqs"])
        pico_freqs = np.array(self.params["pico_freqs"])
        pico_powers = np.array(self.params["pico_powers"])
        (vna_meas,) = self._vna_measurement_labels

        experiment = open_experiment(
            self.save_file_path,
            experiment_name="two_tone_spectro",
            sample_name=Path(self.circuit_path).stem,
        )
        station = Station(self.vna, self.pico, self.yoko)
        run_metadata = instrument_metadata(
            self.instruments, self.params, self.circuit_path
        )

        current_param = Parameter(
            "current", label="DC current", unit="A", get_cmd=None, set_cmd=None
        )
        pico_freq_param = Parameter(
            "pico_frequency",
            label="Anapico frequency",
            unit="Hz",
            get_cmd=None,
            set_cmd=None,
        )
        vna_freq_setting_param = Parameter(
            "vna_frequency_setting",
            label="VNA CW frequency",
            unit="Hz",
            get_cmd=None,
            set_cmd=None,
        )
        pico_power_param = Parameter(
            "pico_power", label="Anapico power", unit="dBm", get_cmd=None, set_cmd=None
        )
        time_param = Parameter("time", label="Timestamp", get_cmd=None, set_cmd=None)
        trace_param = Parameter(
            vna_meas, label=vna_meas, unit="", get_cmd=None, set_cmd=None
        )

        meas = Measurement(name="two_tone_spectro", exp=experiment, station=station)
        meas.register_parameter(current_param, paramtype="numeric")
        meas.register_parameter(pico_freq_param, paramtype="numeric")
        meas.register_parameter(
            vna_freq_setting_param, setpoints=(current_param,), paramtype="numeric"
        )
        meas.register_parameter(
            pico_power_param, setpoints=(pico_freq_param,), paramtype="numeric"
        )
        meas.register_parameter(
            time_param, setpoints=(current_param, pico_freq_param), paramtype="text"
        )
        meas.register_parameter(
            trace_param, setpoints=(current_param, pico_freq_param), paramtype="array"
        )

        # This sweep is the biggest of the four (68017 points / ~69min in
        # the original notebook) with no natural checkpoint, so - like
        # `SpectroDCSweepSlope` - the legacy `.h5` is written cell by cell
        # into the *same* file `BaseMeasurement` already created as each
        # point is measured, not rebuilt/re-read from the db afterwards.
        # The shared 3-D `data` dataset is pre-allocated at its full final
        # size up front (exactly like the old script did), so filling one
        # cell is O(1) regardless of how far into the sweep we are.
        sweep_points = self.vna.points()
        flush_gate = PeriodicFlush(interval=self.params.get("flush_interval", 15.0))

        with safe_run(self.instruments), h5.File(self.save_file_path, "r+") as file:
            data_group = file["data"]
            data_group.create_dataset("Vna frequencies (Hz)", data=vna_freqs)
            data_group.create_dataset("Currents (A)", data=currents)
            data_group.create_dataset("Anapico frequencies (Hz)", data=pico_freqs)
            data_group.create_dataset("Anapico powers (dBm)", data=pico_powers)
            dset = data_group.create_dataset(
                "data",
                (len(currents), len(pico_freqs), sweep_points),
                dtype="complex128",
            )
            dset.dims[0].label = "Currents (A)"
            dset.dims[1].label = "Anapico frequency (Hz)"
            dset.dims[2].label = "CW Time (s)"

            self.pico.output_enabled(True)

            with meas.run() as datasaver:
                for key, value in run_metadata.items():
                    datasaver.dataset.add_metadata(key, value)

                for i_current, current in enumerate(tqdm(currents)):
                    self.yoko.current(current)
                    self.vna.cw(vna_freqs[i_current])
                    for i_pico_freq, pico_freq in enumerate(pico_freqs):
                        self.pico.frequency(pico_freq)
                        self.pico.power(pico_powers[i_pico_freq])
                        self.vna.run_averaging()
                        time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                        trace_data = self.vna.read_raw_data(vna_meas)

                        datasaver.add_result(
                            (current_param, current),
                            (pico_freq_param, pico_freq),
                            (vna_freq_setting_param, vna_freqs[i_current]),
                            (pico_power_param, pico_powers[i_pico_freq]),
                            (time_param, time_str),
                            (trace_param, trace_data),
                        )

                        # Old script overwrote this same dataset-level attr
                        # on every one of the (n_current*n_pico_freq)
                        # writes, so only the very last timestamp ever
                        # survived - reproduced exactly, not fixed into a
                        # per-cell array, to match the legacy format.
                        dset[i_current, i_pico_freq] = trace_data
                        dset.attrs["time"] = time_str

                        if flush_gate.due():
                            file.flush()
                            datasaver.flush_data_to_database(block=False)

            file.flush()
            self.pico.output_enabled(False)
