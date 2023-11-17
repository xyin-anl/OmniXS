# %%
import numpy as np
from scipy.stats import cauchy
from src.raw_data_vasp import RAWDataVASP


class VASPDataModifier:
    def __init__(self, spectra_params):
        self.parameters = spectra_params  # for single spectra
        self.parsed_data = self.parameters["mu"]
        self.energy_full = self.parsed_data[:, 0]
        self.spectra_full = self.parsed_data[:, 1]
        self.e_core = self.parameters["e_core"]
        self.e_cbm = self.parameters["e_cbm"]
        self.E_ch = self.parameters["E_ch"]
        self.E_GS = self.parameters["E_GS"]
        self.volume = self.parameters["volume"]
        self.start_offset = 5
        self.end_offset = 5
        self.energy, self.spectra = None, None
        # self.Gamma = 0.89
        # more attributes inside functions for plot
        self.transform()

    def transform(self):
        self.energy, self.spectra = self.truncate()
        self.spectra = self.scale()
        self.spectra = self.broaden(gamma=0.89)
        self.energy = self.align(experimental_shift=5114.08973)
        return self

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
        return energy, spectra

    # def truncate(self):
    #     minimum_spectra = np.median(self.spectra_full) * 0.01
    #     minimum_energy = (self.e_cbm - self.e_core) - self.start_offset
    #     min_idx = np.where(self.energy_full > minimum_energy)[0][0]
    #     energy, spectra = self.energy_full[min_idx:], self.spectra_full[min_idx:]
    #     min_idx = np.where(spectra > minimum_spectra)[0][0]
    #     energy, spectra = energy[min_idx:], spectra[min_idx:]
    #     max_idx = np.where(spectra > minimum_spectra)[0][-1]
    #     energy, spectra = energy[:max_idx], spectra[:max_idx]
    #     max_idx = np.where(energy < energy[-1] - self.end_offset)[0][-1]
    #     energy, spectra = energy[:max_idx], spectra[:max_idx]
    #     self.energy_trunc, self.spectra_trunc = energy, spectra  # plot
    #     return energy, spectra

    def scale(self):
        omega = self.spectra * self.energy
        omega /= 13.6056980659 * 2  # eV to Hartree
        self.big_omega = self.volume
        self.alpha = 137.036
        spectra_scaled = (omega * self.big_omega) / self.alpha
        self.spectra_scaled = spectra_scaled  # for plot
        return spectra_scaled

    @classmethod
    def lorentz_broaden(self, x, xin, yin, gamma):
        dx = xin[-1] - xin[0]
        differences = x[:, np.newaxis] - xin
        lorentzian = cauchy.pdf(differences, 0, gamma / 2)
        return np.dot(lorentzian, yin) / len(xin) * dx

    # @classmethod
    # def lorentz_broaden(self, x, xin, yin, gamma):
    #     x1, x2 = np.meshgrid(x, xin)
    #     dx = xin[-1] - xin[0]
    #     return np.dot(cauchy.pdf(x1, x2, gamma / 2).T, yin) / len(xin) * dx

    def broaden(self, gamma=0.89):
        # gamma = self.Gamma / 2
        broadened_amplitude = self.lorentz_broaden(
            self.energy, self.energy, self.spectra_scaled, gamma=gamma
        )
        self.spectra_boardened = broadened_amplitude  # for plot
        return broadened_amplitude

    def align(self, experimental_shift=5114.08973):
        self.align_offset = (self.e_core - self.e_cbm) + (self.E_ch - self.E_GS)
        energy_aligned = self.energy + self.align_offset
        self.energy_aligned = energy_aligned  # for plot
        self.energy_aligned += experimental_shift
        return energy_aligned


if __name__ == "__main__":
    compound = "Ti"
    simulation_type = "VASP"
    data = RAWDataVASP(compound, simulation_type)

    # id = next(iter(data.parameters))
    id = ("mp-390", "000_Ti")  # reference to another paper data
    transform = VASPDataModifier(data.parameters[id])

    from matplotlib import pyplot as plt

    plt.plot(transform.energy, transform.spectra)
    plt.plot(transform.energy, transform.spectra_scaled, alpha=0.5)
    plt.show()
