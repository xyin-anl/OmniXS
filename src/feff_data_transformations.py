# %%
import numpy as np
from scipy.stats import cauchy
from src.vasp_data_transformations import VASPDataModifier


class FEFFDataModifier:
    def __init__(self, spectra_params):
        self.parameters = spectra_params  # for single spectra
        self.energy_raw = self.parameters["mu"][:, 0]
        self.spectra_raw = self.parameters["mu"][:, 1]
        self.start_offset = 5
        self.end_offset = 5
        self.energy, self.spectra = None, None
        self.transform()

    def transform(self):
        self.energy, self.spectra = self.truncate()
        self.spectra = self.broaden()

    def truncate(self):
        minimum_spectra = np.median(self.spectra_raw) * 0.01
        spectra, energy = self.spectra_raw, self.energy_raw
        min_idx = np.where(spectra > minimum_spectra)[0][0]
        energy, spectra = energy[min_idx:], spectra[min_idx:]
        max_idx = np.where(spectra > minimum_spectra)[0][-1]
        energy, spectra = energy[:max_idx], spectra[:max_idx]
        max_idx = np.where(energy < energy[-1] - self.end_offset)[0][-1]
        energy, spectra = energy[:max_idx], spectra[:max_idx]
        self.energy_trunc, self.spectra_trunc = energy, spectra  # plot
        return energy, spectra

    def broaden(self, gamma=0.89 / 2):
        broadened_amplitude = VASPDataModifier.lorentz_broaden(
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
