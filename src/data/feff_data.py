import numpy as np
from src.data.data import ProcessedData
from src.lightshow.compare_utils import compare_between_spectra
from scipy import constants


class FEFFData(ProcessedData):
    def __init__(
        self,
        spectra_params,
        transform=True,
    ):
        super().__init__(spectra_params, transform=transform)

    def transform(self):
        return self.truncate().scale()

    def truncate(self):
        return self.filter(spectral_range=[0, None])

    def scale(self):
        bohrs_radius = constants.physical_constants["Bohr radius"][0]
        bohrs_radius /= constants.angstrom
        self._spectra = self.spectra / (bohrs_radius**2)
        return self

    def align(self, target: ProcessedData):
        source = np.array([self.energy, self.spectra]).T
        target = np.array([target.energy, target.spectra]).T
        shift, _ = compare_between_spectra(source, target)
        return self.align_energy(shift)


if __name__ == "__main__":
    from src.data.feff_data_raw import RAWDataFEFF

    feff_raw = RAWDataFEFF(compound="Ti")
    feff_transformed = FEFFData(feff_raw.parameters[("mp-390", "000_Ti")])

    from matplotlib import pyplot as plt

    plt.plot(feff_transformed.energy, feff_transformed.spectra)
    plt.show()
