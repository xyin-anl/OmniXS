import numpy as np
from src.data.data_transformations import DataModifier
from src.lightshow.compare_utils import compare_between_spectra
from scipy import constants


class FEFFDataModifier(DataModifier):
    def __init__(
        self,
        spectra_params,
        transform=True,
    ):
        super().__init__(spectra_params, transform=transform)

    def transform(self):
        return self.truncate().scale().align()

    def truncate(self):
        return self.filter(spectral_range=[0, None])

    def scale(self):
        bohrs_radius = constants.physical_constants["Bohr radius"][0]
        bohrs_radius /= constants.angstrom
        self._spectra = self.spectra / (bohrs_radius**2)
        return self

    def spearman_align(self, target_spectra):
        """useful to align feff with vasp spectra"""

        # helper function
        def obj_to_array(obj):
            return np.array([obj.energy, obj.spectra]).T

        spectra_1 = obj_to_array(self)
        target_spectra = obj_to_array(target_spectra)
        shift, _ = compare_between_spectra(spectra_1, target_spectra)
        shift *= 1  #  feff needs to be shifted by this
        aligned_energy = self.energy - shift
        self._energy = aligned_energy
        # return self
        return shift


if __name__ == "__main__":
    from src.data.raw_data_feff import RAWDataFEFF

    feff_raw = RAWDataFEFF(compound="Ti")
    feff_transformed = FEFFDataModifier(feff_raw.parameters[("mp-390", "000_Ti")])

    from matplotlib import pyplot as plt

    plt.plot(feff_transformed.energy, feff_transformed.spectra)
    plt.show()
