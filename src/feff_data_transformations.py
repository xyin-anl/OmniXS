# %%
import numpy as np
from scipy.stats import cauchy


class FEFFDataModifier:
    def __init__(self, spectra_params):
        self.parameters = spectra_params  # for single spectra
        self.parsed_data = self.parameters["mu"]
        self.energy_full = self.parsed_data[:, 0]
        self.spectra_full = self.parsed_data[:, 1]
        self.start_offset = 5
        self.end_offset = 5
        self.energy, self.spectra = None, None
        self.Gamma = 0.89
        self.transform()

    def transform(self):
        self.energy, self.spectra = self.truncate()
        self.spectra = self.broaden()

    def truncate(self):
        minimum_spectra = np.median(self.spectra_full) * 0.01
        # minimum_energy = 0  # TODO: remove this later
        # min_idx = np.where(self.energy_full > minimum_energy)[0][0]
        # energy, spectra = self.energy_full[min_idx:], self.spectra_full[min_idx:]
        spectra, energy = self.spectra_full, self.energy_full
        min_idx = np.where(spectra > minimum_spectra)[0][0]
        energy, spectra = energy[min_idx:], spectra[min_idx:]
        max_idx = np.where(spectra > minimum_spectra)[0][-1]
        energy, spectra = energy[:max_idx], spectra[:max_idx]
        max_idx = np.where(energy < energy[-1] - self.end_offset)[0][-1]
        energy, spectra = energy[:max_idx], spectra[:max_idx]
        self.energy_trunc, self.spectra_trunc = energy, spectra  # plot
        return energy, spectra

    def _lorentz_broaden(self, x, xin, yin, gamma):
        x1, x2 = np.meshgrid(x, xin)
        dx = xin[-1] - xin[0]
        return np.dot(cauchy.pdf(x1, x2, gamma).T, yin) / len(xin) * dx

    def broaden(self):
        gamma = self.Gamma / 2
        broadened_amplitude = self._lorentz_broaden(
            self.energy,
            self.energy,
            self.spectra,
            gamma=gamma,
        )
        self.broadened_amplitude = broadened_amplitude  # for plot
        return broadened_amplitude

    def scale(self):
        pass

    def align(self):
        pass


if __name__ == "__main__":
    from src.raw_data_feff import RAWDataFEFF

    feff = RAWDataFEFF(compound="Ti")
    transform = FEFFDataModifier(feff.parameters[("mp-390", "000_Ti")])

    from matplotlib import pyplot as plt

    plt.plot(transform.energy, transform.spectra)
    plt.show()

# %%
