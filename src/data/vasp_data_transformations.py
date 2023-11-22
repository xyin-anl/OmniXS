from scipy.constants import physical_constants
import warnings
import numpy as np
from scipy.stats import cauchy
from src.data.raw_data_vasp import RAWDataVASP
from src.data.data_transformations import DataModifier


class VASPDataModifier(DataModifier):
    # EXPERIMENTAL_ENERGY_OFFSET = 5114.08973  # based on comparison with xs_mp_390

    def __init__(self, spectra_params, transform=True):
        self.e_core = spectra_params["e_core"]
        self.e_cbm = spectra_params["e_cbm"]
        self.E_ch = spectra_params["E_ch"]
        self.E_GS = spectra_params["E_GS"]
        self.volume = spectra_params["volume"]
        super().__init__(spectra_params, transform=transform)

    def transform(self):
        """Apply truncation, scaling, broadening, and alignment."""
        return self.truncate().scale().broaden().align()

    def truncate(self, start_offset=10):
        min_energy = (self.e_cbm - self.e_core) - start_offset
        self.filter(energy_range=[min_energy, None])
        self.filter(spectral_range=[0, None])
        return self

    def scale(self):
        r = physical_constants["Rydberg constant times hc in eV"][0]
        a = physical_constants["inverse fine-structure constant"][0]
        omega = self._spectra * self._energy
        omega /= r * 2
        self.big_omega = self.volume
        self._spectra = (omega * self.big_omega) / a
        return self

    @classmethod
    def lorentz_broaden(self, x, xin, yin, gamma):
        dx = xin[-1] - xin[0]
        differences = x[:, np.newaxis] - xin
        lorentzian = cauchy.pdf(differences, 0, gamma / 2)
        return np.dot(lorentzian, yin) / len(xin) * dx

    def broaden(self, gamma=0.89):
        broadened_amplitude = self.lorentz_broaden(
            self._energy,
            self._energy,
            self._spectra,
            gamma=gamma,
        )
        self._spectra = broadened_amplitude
        return self

    def align(self, emperical_offset=5114.08973):
        thoeretical_offset = (self.e_core - self.e_cbm) + (self.E_ch - self.E_GS)
        super().align(thoeretical_offset)
        super().align(emperical_offset)
        return self


if __name__ == "__main__":
    compound = "Ti"
    simulation_type = "VASP"
    data = RAWDataVASP(compound, simulation_type)

    id = ("mp-390", "000_Ti")  # reference to another paper data
    data = VASPDataModifier(data.parameters[id])

    from matplotlib import pyplot as plt
    import scienceplots

    plt.style.use(["default", "science"])
    fig = plt.figure(figsize=(8, 6))
    data.truncate()
    plt.plot(data.energy, data.spectra, label="truncated")
    data.scale()
    plt.plot(data.energy, data.spectra, label="trucated and scaled")
    data.broaden()
    plt.plot(data.energy, data.spectra, label="truncated, scaled, and broadened")
    data.align()
    plt.plot(data.energy, data.spectra, label="truncated, scaled, broadened, aligned")
    plt.xlabel("Energy (eV)")
    plt.legend()
    plt.title(f"VASP spectra for {id}")
    # plt.savefig("vasp_transformations.pdf", bbox_inches="tight", dpi=300)
    plt.show()
