import warnings
from typing import Union
import os
import pickle
from functools import cached_property
from typing import List

import appdirs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from p_tqdm import p_map

from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data import VASPData
from src.data.vasp_data_raw import RAWDataVASP
from utils.src.plots.heatmap_of_lines import heatmap_of_lines


class SpectraTable:
    """Constains the all of processed spectra"""

    def __init__(
        self,
        compound: str,
        simulation_type: str,
        use_cache: bool = False,
        data: Union[Union[List[FEFFData], List[VASPData]], None] = None,
    ):

        self._check_input_validity(compound, simulation_type, data)
        self.compound = compound
        self.simulation_type = simulation_type
        if data is not None:
            self._data = data
        else:
            self._data = None  # to avoid circular reference in cached_property
            if use_cache:
                self._data = self._get_cache()
            else:
                # just remove cache just to be safe
                cache_dir = appdirs.user_cache_dir("xas_ml")
                file_name = f"SpectraTable_{self.compound}_{self.simulation_type}.pkl"
                file_path = os.path.join(cache_dir, file_name)
                if os.path.exists(file_path):
                    warnings.warn(f"Cache file exists. Removing {file_path}")
                    os.remove(file_path)
                self._data = self.data

    @cached_property
    def data(self):
        if self._data is not None:
            return self._data
        data_class = FEFFData if self.simulation_type == "FEFF" else VASPData
        map_fn = lambda id: data_class(self.compound, self.parameters[id], id)
        return p_map(map_fn, self.ids)

    @property
    def ids(self):
        return self.raw_data.ids

    @property
    def parameters(self):
        return self.raw_data.parameters

    @cached_property
    def raw_data(self):
        raw_class = RAWDataFEFF if self.simulation_type == "FEFF" else RAWDataVASP
        return raw_class(self.compound)

    def resample(self):
        self._data = list(map(lambda x: x.resample().spectra, self._data))
        return self

    def _check_input_validity(self, compound, simulation_type, data):
        if data is None:
            return
        expected_cls = FEFFData if simulation_type == "FEFF" else VASPData
        expected_sims = ["FEFF", "VASP"]
        compound_check = all([x.compound == compound for x in data])

        cls_check = all([isinstance(x, expected_cls) for x in data])
        sim_check = all([x.simulation_type in expected_sims for x in data])
        checks = [cls_check, sim_check, compound_check]
        passed = all(checks)

        err_msg = "Data does not match expected format"
        raise ValueError(err_msg) if not passed else None

    def _get_cache(self):
        cache_dir = appdirs.user_cache_dir("xas_ml")
        file_name = f"SpectraTable_{self.compound}_{self.simulation_type}.pkl"
        file_path = os.path.join(cache_dir, file_name)
        if os.path.exists(file_path):
            return pickle.load(open(file_path, "rb"))
        else:
            data = self.data
            os.makedirs(cache_dir, exist_ok=True)

            pickle.dump(data, open(file_path, "wb"))
            return data

    def save(self, file_path: str = "./"):
        for spectra in self.data:
            spectra.save(file_path)
        return self

    def plot(self, ax: plt.Axes = None):
        heatmap_of_lines(np.array([x.spectra for x in self.data]), ax=ax)
        return self


if __name__ == "__main__":
    SpectraTable("Cu", "VASP", use_cache=True).resample().save()
