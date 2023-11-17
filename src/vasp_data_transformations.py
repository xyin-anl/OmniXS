from scipy import constants
import warnings
import numpy as np
from scipy.stats import cauchy
from src.raw_data_vasp import RAWDataVASP


class VASPDataModifier:
    EXPERIMENTAL_ENERGY_OFFSET = 5114.08973  # based on comparison with xs_mp_390

    def __init__(self, spectra_params, transform=True):
        self.parameters = spectra_params  # for single spectra
        self.parsed_data = self.parameters["mu"]
        self.energy_full = self.parsed_data[:, 0]
        self.spectra_full = self.parsed_data[:, 1]
        self.e_core = self.parameters["e_core"]
        self.e_cbm = self.parameters["e_cbm"]
        self.E_ch = self.parameters["E_ch"]
        self.E_GS = self.parameters["E_GS"]
        self.volume = self.parameters["volume"]
        self.start_offset = 0  # TODO: remove this emperical value
        self.end_offset = 0
        self._energy, self._spectra = None, None
        if transform:
            self.transform()

    def reset(self):
        self._energy, self._spectra = None, None
        return self

    @property
    def energy(self):
        if self._energy is None:
            warn_text = "Data is not transformed yet. \n"
            warn_text += "Applying truncation, scaling, broadening, and alignment."
            warnings.warn(warn_text)
            self.transform()
        return self._energy

    @property
    def spectra(self):
        if self._spectra is None:
            warn_text = "Data is not transformed yet. \n"
            warn_text += "Applying truncation, scaling, broadening, and alignment."
            warnings.warn(warn_text)
            self.transform()
        return self._spectra

    def transform(self):
        """Apply truncation, scaling, broadening, and alignment."""
        return self.truncate().scale().broaden().align()

    def truncate(self):
        minimum_spectra = np.median(self.spectra_full) * 0.01
        minimum_energy = (self.e_cbm - self.e_core) - self.start_offset
        valid_energy = self.energy_full > minimum_energy
        valid_spectra = self.spectra_full[valid_energy] > minimum_spectra
        energy, spectra = (
            self.energy_full[valid_energy][valid_spectra],
            self.spectra_full[valid_energy][valid_spectra],
        )
        max_energy_index = np.where(energy < energy[-1] - self.end_offset)[0][-1]
        energy, spectra = energy[:max_energy_index], spectra[:max_energy_index]
        self.energy_trunc, self.spectra_trunc = energy, spectra
        self._energy, self._spectra = energy, spectra
        return self

    def scale(self):
        omega = self._spectra * self._energy
        omega /= 13.6056980659 * 2  # eV to Hartree
        self.big_omega = self.volume
        self.alpha = 137.036
        spectra_scaled = (omega * self.big_omega) / self.alpha
        self.spectra_scaled = spectra_scaled  # for plot
        self._spectra = spectra_scaled
        return self

    @classmethod
    def lorentz_broaden(self, x, xin, yin, gamma):
        dx = xin[-1] - xin[0]
        differences = x[:, np.newaxis] - xin
        lorentzian = cauchy.pdf(differences, 0, gamma / 2)
        return np.dot(lorentzian, yin) / len(xin) * dx

    def broaden(self, gamma=0.89):
        broadened_amplitude = self.lorentz_broaden(
            self._energy, self._energy, self.spectra_scaled, gamma=gamma
        )
        self.spectra_boardened = broadened_amplitude  # for plot
        self._spectra = broadened_amplitude
        return self

    def align(self, energy_offset=EXPERIMENTAL_ENERGY_OFFSET):
        self.align_offset = (self.e_core - self.e_cbm) + (self.E_ch - self.E_GS)
        energy_aligned = self._energy + self.align_offset
        self.energy_aligned = energy_aligned  # for plot
        self.energy_aligned += energy_offset
        self._energy = energy_aligned
        return self


if __name__ == "__main__":
    compound = "Ti"
    simulation_type = "VASP"
    data = RAWDataVASP(compound, simulation_type)

    id = ("mp-390", "000_Ti")  # reference to another paper data
    transform = VASPDataModifier(data.parameters[id])

    from matplotlib import pyplot as plt

    plt.plot(transform._energy, transform._spectra)
    plt.plot(transform._energy, transform.spectra_scaled, alpha=0.5)
    plt.show()
