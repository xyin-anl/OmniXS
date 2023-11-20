import numpy as np
from src.compare_utils import compare_between_spectra
from scipy import constants


class FEFFDataModifier:
    def __init__(
        self,
        spectra_params,
        transform=True,
        # emperical_energy_offset=-1.24,  # 1.24 is for Ti (median)
        emperical_energy_offset=-0.6,  # 0.6 is for Ti (mean)
    ):
        self.parameters = spectra_params  # for single spectra
        self.energy_full = self.parameters["mu"][:, 0]
        self.spectra_full = self.parameters["mu"][:, 1]
        self._energy, self._spectra = None, None
        self.emperical_energy_offset = emperical_energy_offset
        if transform:
            self.transform()

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

    def filter(self, energy_range):
        """Filter energy and spectra based on energy range"""
        energy_filter = (self.energy >= energy_range[0]) & (
            self.energy <= energy_range[1]
        )
        self._energy = self.energy[energy_filter]
        self._spectra = self.spectra[energy_filter]
        return self

    def __repr__(self):
        string = "FeFF data post transformations:\n"
        string += f"energy: {self.energy}\n"
        string += f"spectra: {self.spectra}\n"
        return string

    def reset(self):
        self._energy, self._spectra = None, None
        return self

    def transform(self):
        return self.truncate().scale().align()

    # TODO: change median_fraction
    def truncate(self, median_fraction=0.01, start_offset=0, end_offset=0):
        # chop energy in from where spectra is low (small fraction of median)
        # and provided offset
        minimum_spectra = np.median(self.spectra_full) * median_fraction - start_offset
        min_idx = np.where(self.spectra_full > minimum_spectra)[0][0]
        energy, spectra = self.energy_full[min_idx:], self.spectra_full[min_idx:]
        max_idx = np.where(spectra > minimum_spectra)[0][-1]
        energy, spectra = energy[:max_idx], spectra[:max_idx]
        # chop again at end based of provided end_offset
        max_idx = np.where(energy < energy[-1] - end_offset)[0][-1]
        energy, spectra = energy[:max_idx], spectra[:max_idx]
        self._energy, self._spectra = energy, spectra
        return self

    def spearman_align(self, target_spectra):
        """useful to align feff with vasp spectra"""

        # helper function
        def obj_to_array(obj):
            return np.array([obj.energy, obj.spectra]).T

        spectra_1 = obj_to_array(self)
        target_spectra = obj_to_array(target_spectra)
        shift, _ = compare_between_spectra(spectra_1, target_spectra)
        return shift
        # aligned_energy = self.energy - shift
        # self._energy = aligned_energy
        # return self

    def scale(self):
        # theoretical
        bohrs_radius = constants.physical_constants["Bohr radius"][0]
        bohrs_radius /= constants.angstrom
        self._spectra = self.spectra / (bohrs_radius**2)
        return self

    def align(self):
        # self._energy = self.energy - self.emperical_energy_offset
        return self


if __name__ == "__main__":
    from src.raw_data_feff import RAWDataFEFF

    feff = RAWDataFEFF(compound="Ti")
    feff_transformed = FEFFDataModifier(feff.parameters[("mp-390", "000_Ti")])
    feff_transformed.transform()

    from matplotlib import pyplot as plt

    plt.plot(feff_transformed.energy, feff_transformed.spectra)
    plt.show()
