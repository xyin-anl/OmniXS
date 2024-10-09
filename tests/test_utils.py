from matgl.ext.pymatgen import Structure

from refactor.spectra_data import Material, Site, Spectrum


def create_dummy_structure():
    return Structure(
        lattice=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], species=["H"], coords=[[0, 0, 0]]
    )


def create_dummy_site():
    return Site(
        index=0,
        element="H",
        spectra={"FEFF": create_dummy_spectrum()},
    )


def create_dummy_spectrum():
    return Spectrum(type="FEFF", energies=[1, 2, 3], intensities=[4, 5, 6])


def create_dummy_material(id="mp-1234"):
    return Material(id=id, structure=create_dummy_structure())
