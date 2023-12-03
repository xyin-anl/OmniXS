import yaml
import numpy as np
from dtw import dtw
import os


import warnings
from abc import ABC, abstractmethod


class ProcessedData(ABC):
    def __init__(self, compound, simulation_type, params=None, id=None):
        self.compound = compound
        self.simulation_type = simulation_type
        self._id = id
        self._energy = self._spectra = None

        if params is not None:
            self.params = params
            self._energy = self.energy_full = self.params["mu"][:, 0]
            self._spectra = self.spectra_full = self.params["mu"][:, 1]
            self.transform()

    @property
    def id(self):
        if self._id is None:
            raise ValueError("id not set")
        return self._id

    @id.setter
    def id(self, id):
        assert (
            isinstance(id, tuple)
            and len(id) == 2
            and all(isinstance(i, str) for i in id)
        ), "id must be a tuple of two strings"
        self._id = id

    def load(self, id, file_path=None):
        if self._energy is not None or self._spectra is not None:
            warnings.warn("Data already loaded. Overwriting.")
        self.id = id
        if file_path is None:
            file_path = os.path.join(
                "dataset",
                f"{self.simulation_type}-processed-data",
                self.compound,
                "_site_".join(self.id) + ".dat",
            )
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        data = np.loadtxt(file_path)
        self._energy = self.energy_full = data[:, 0]
        self._spectra = self.spectra_full = data[:, 1]
        return self

    @property
    def energy(self):
        if self._energy is None:
            raise ValueError("Energy values empty. Load data first.")
        return self._energy

    @property
    def spectra(self):
        if self._spectra is None:
            raise ValueError("Spectra values empty. Load data first.")
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

    @staticmethod
    def dtw_shift(source: "ProcessedData", target: "ProcessedData"):
        # 4x faster than compare_between_spectra
        # not consistently better compare_between_spectra
        # sometime worse
        d, cost_matrix, acc_cost_matrix, path = dtw(
            source.spectra,
            target.spectra,
            dist=lambda x, y: np.abs(x - y),  # euclidean norm
        )
        shifts = source.energy[path[0]] - target.energy[path[1]]
        dominant_shift = np.round(np.median(shifts)).astype(int)
        return dominant_shift

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

    def save(self, save_dir="."):
        save_dir = os.path.join(
            save_dir,
            f"{self.simulation_type}-processed-data",
            self.compound,
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            warnings.warn(f"Created directory {save_dir} to save data.")
        file_name = "_site_".join(self._id) + ".dat"
        file_path = os.path.join(save_dir, file_name)
        data_table = np.array([self.energy, self.spectra]).T
        np.savetxt(file_path, data_table, delimiter="\t")
