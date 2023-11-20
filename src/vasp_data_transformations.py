from scipy.constants import physical_constants
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
        # self.start_offset = 0  # TODO: remove this emperical value
        # self.end_offset = 0
        self._energy, self._spectra = None, None
        if transform:
            self.transform()

    def reset(self):
        self._energy, self._spectra = None, None
        return self

    def filter(self, energy_range):
        """Filter energy and spectra based on energy range"""
        energy_filter = (self.energy > energy_range[0]) & (
            self.energy < energy_range[1]
        )
        self._energy = self.energy[energy_filter]
        self._spectra = self.spectra[energy_filter]
        return self

    def __repr__(self):
        string = "VASP data post transformations:\n"
        string += f"energy: {self.energy}\n"
        string += f"spectra: {self.spectra}\n"
        return string

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

    def truncate(self, median_fraction=0.01, start_offset=0, end_offset=0):
        """Truncate the energy and spectra based on theory, emperical offset
        and low spectra values"""
        # chop on front based on theory and any provided offset
        minimum_energy = (self.e_cbm - self.e_core) - start_offset
        energy_filter = self.energy_full > minimum_energy
        # remove energy values with low spectra values based on median
        minimum_spectra = np.median(self.spectra_full) * median_fraction
        spectra_filter = self.spectra_full[energy_filter] > minimum_spectra
        energy, spectra = (
            self.energy_full[energy_filter][spectra_filter],
            self.spectra_full[energy_filter][spectra_filter],
        )
        # chop end of energy if any offset is provided
        max_energy_index = np.where(energy < energy[-1] - end_offset)[0][-1]
        energy, spectra = energy[:max_energy_index], spectra[:max_energy_index]
        self._energy, self._spectra = energy, spectra
        return self

    def scale(self):
        """Scale spectra bsed on theory"""
        omega = self._spectra * self._energy
        rydbrg_constant = physical_constants["Rydberg constant times hc in eV"][0]
        omega /= rydbrg_constant * 2
        self.big_omega = self.volume
        alpha = physical_constants["inverse fine-structure constant"][0]
        spectra_scaled = (omega * self.big_omega) / alpha
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
            self._energy,
            self._energy,
            self._spectra,
            gamma=gamma,
        )
        self._spectra = broadened_amplitude
        return self

    def align(self, emperical_offset=EXPERIMENTAL_ENERGY_OFFSET):
        """Align energy based on theory and emperical offset"""
        self.thoeretical_offset = (self.e_core - self.e_cbm) + (self.E_ch - self.E_GS)
        energy_aligned = self._energy + self.thoeretical_offset
        energy_aligned += emperical_offset
        self._energy = energy_aligned
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
