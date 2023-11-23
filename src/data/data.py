import yaml
import numpy as np


import warnings
from abc import ABC, abstractmethod


class ProcessedData(ABC):
    def __init__(self, compound, simulation_type, params):
        self.compound = compound
        self.simulation_type = simulation_type
        self.params = params
        self._energy = self.energy_full = self.params["mu"][:, 0]
        self._spectra = self.spectra_full = self.params["mu"][:, 1]
        self.transform()

    @property
    def energy(self):
        return self._energy

    @property
    def spectra(self):
        return self._spectra

    def reset(self):
        self._energy, self._spectra = self.energy_full, self.spectra_full
        return self

    def truncate_emperically(self):
        cfg = self.configs()[self.simulation_type][self.compound]
        e_range = cfg["e_range"]
        return self.filter(energy_range=e_range)

    def filter(self, energy_range=(None, None), spectral_range=(None, None)):
        e_start, e_end = energy_range
        s_start, s_end = spectral_range

        energy, spectra = self._energy, self._spectra
        e_filter = True if e_start is None else energy > e_start
        e_filter &= True if e_end is None else energy < e_end
        s_filter = True if s_start is None else spectra > s_start
        s_filter &= True if s_end is None else spectra < s_end

        all_filters = e_filter & s_filter
        indices = np.where(all_filters)[0]
        if len(indices) == 0:
            warnings.warn("No data points are left after filtering.")
            return self
        min_idx, max_idx = indices[0], indices[-1] + 1
        self._energy = self._energy[min_idx:max_idx]
        self._spectra = self._spectra[min_idx:max_idx]
        return self

    def __repr__(self):
        string = "Data post transformations:\n"
        string += f"energy: {self.energy}\n"
        string += f"spectra: {self.spectra}\n"
        return string

    def configs(self, cfg_path="cfg/transformations.yaml"):
        with open(cfg_path) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        return cfg

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def truncate(self):
        pass

    @abstractmethod
    def scale(self):
        pass

    def align_energy(self, energy_offset=0):
        self._energy = self._energy + energy_offset
        return self
