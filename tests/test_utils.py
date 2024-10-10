from matgl.ext.pymatgen import Structure

from refactor.spectra_data import Material, AbsorptionSite, Spectrum
from refactor.spectra_enums import Element, SpectrumType


def create_dummy_structure():
    return Structure(
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], species=["H"], coords=[[0, 0, 0]]
    )
