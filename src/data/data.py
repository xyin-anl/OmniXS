import numpy as np


import warnings
from abc import ABC, abstractmethod


class ProcessedData(ABC):
    def __init__(self, spectra_params, transform=True):
        self.parameters = spectra_params
        self.energy_full = self.parameters["mu"][:, 0]
        self.spectra_full = self.parameters["mu"][:, 1]
        self._energy, self._spectra = None, None
        if transform:
            self.transform()

    @property
    def energy(self):
        if self._energy is None:
            warn_text = "Data is not transformed yet. \n"
            warn_text += "Applying transformations."
            warnings.warn(warn_text)
            self.transform()
        return self._energy

    @property
    def spectra(self):
        if self._spectra is None:
            warn_text = "Data is not transformed yet."
            warn_text += "Call transform() first."
            warnings.warn(warn_text)
            self.transform()
        return self._spectra

    def reset(self):
        self._energy, self._spectra = None, None
        return self

    def filter(self, energy_range=(None, None), spectral_range=(None, None)):
        e_start, e_end = energy_range
        s_start, s_end = spectral_range

        if self._energy is None or self._spectra is None:
            self._energy, self._spectra = self.energy_full, self.spectra_full
        energy, spectra = self._energy, self._spectra

        e_filter = True if e_start is None else energy > e_start
        e_filter &= True if e_end is None else energy < e_end
        s_filter = True if s_start is None else spectra > s_start
        s_filter &= True if s_end is None else spectra < s_end

        all_filters = e_filter & s_filter
        indices = np.where(all_filters)[0]
        if len(indices) == 0:
            raise ValueError("No data points are left after filtering.")
        min_idx, max_idx = indices[0], indices[-1] + 1
        self._energy = self._energy[min_idx:max_idx]
        self._spectra = self._spectra[min_idx:max_idx]
        return self

    def __repr__(self):
        string = "Data post transformations:\n"
        string += f"energy: {self.energy}\n"
        string += f"spectra: {self.spectra}\n"
        return string

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
