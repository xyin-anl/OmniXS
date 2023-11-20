import numpy as np
from src.compare_utils import compare_between_spectra
from scipy import constants


class FEFFDataModifier:
    EMPERICAL_ENERGY_OFFSET = 0  # TODO: add this after data processing

    def __init__(self, spectra_params):
        self.parameters = spectra_params  # for single spectra
        self.energy_raw = self.parameters["mu"][:, 0]
        self.spectra_raw = self.parameters["mu"][:, 1]
        self.start_offset = 5
        self.end_offset = 5
        self._energy, self._spectra = None, None

    @property
    def energy(self):
        if self._energy is None:
            err_text = "Data is not transformed yet."
            err_text += "Call truncate() and/or align() first."
            raise ValueError(err_text)
        return self._energy

    @property
    def spectra(self):
        if self._spectra is None:
            err_text = "Data is not transformed yet."
            err_text += "Call truncate() and/or align() first."
            raise ValueError(err_text)
        return self._spectra

    def reset(self):
        self._energy, self._spectra = None, None
        return self

    def transform(self):
        return self.truncate().align()

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
        self._energy, self._spectra = energy, spectra
        return self

    def align_to_spectra(self, target_spectra):
        # useful to align feff with vasp spectra
        def obj_to_array(obj):
            return np.array([obj.energy, obj.spectra]).T

        spectra_1 = obj_to_array(self)
        target_spectra = obj_to_array(target_spectra)
        shift, _ = compare_between_spectra(spectra_1, target_spectra)
        aligned_energy = self.energy - shift
        self._energy = aligned_energy
        return self

    def scale(self):
        bohrs_radius = constants.physical_constants["Bohr radius"][0]
        bohrs_radius /= constants.angstrom
        self._spectra = self.spectra / (bohrs_radius**2)
        return self

    def align(self):
        self._energy = self.energy + self.EMPERICAL_ENERGY_OFFSET
        return self


if __name__ == "__main__":
    from src.raw_data_feff import RAWDataFEFF

    feff = RAWDataFEFF(compound="Ti")
    feff_transformed = FEFFDataModifier(feff.parameters[("mp-390", "000_Ti")])
    feff_transformed.transform()

    from matplotlib import pyplot as plt

    plt.plot(feff_transformed.energy, feff_transformed.spectra)
    plt.show()
