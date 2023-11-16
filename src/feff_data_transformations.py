import numpy as np
from src.compare_utils import compare_between_spectra


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

    def align_to_spectra(self, target_spectra):
        # useful to align feff with vasp spectra
        def obj_to_array(obj):
            return np.array([obj.energy, obj.spectra]).T

        spectra_1 = obj_to_array(self)
        target_spectra = obj_to_array(target_spectra)
        shift, _ = compare_between_spectra(spectra_1, target_spectra)
        aligned_energy = self.energy - shift
        self.energy = aligned_energy
        return self

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
