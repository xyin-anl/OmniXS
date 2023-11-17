import matplotlib.pyplot as plt
import numpy as np

from src.feff_data_transformations import FEFFDataModifier
from src.raw_data_feff import RAWDataFEFF
from src.raw_data_vasp import RAWDataVASP
from src.vasp_data_transformations import VASPDataModifier
import scienceplots

if __name__ == "__main__":
    compound = "Ti"
    id = ("mp-390", "000_Ti")
    vasp = RAWDataVASP(compound=compound)
    vasp_spectra = VASPDataModifier(vasp.parameters[id])

    vasp_spectra.energy, vasp_spectra.spectra = None, None
    vasp_spectra.energy, vasp_spectra.spectra = vasp_spectra.truncate()
    vasp_spectra.spectra = vasp_spectra.scale()
    vasp_spectra.spectra = vasp_spectra.broaden(gamma=0.89 * 2)
    vasp_spectra.energy = vasp_spectra.align(experimental_shift=5114.08973)

    feff = RAWDataFEFF(compound=compound)
    feff_spectra = FEFFDataModifier(feff.parameters[id])
    feff_spectra.align_to_spectra(vasp_spectra)

    xs_mp_390 = np.load("dataset/misc/xs-mp-390.npy")
    xs_energy, xs_spectra = xs_mp_390[0], xs_mp_390[1]
    xs_energy -= (
        xs_energy[np.argmax(xs_spectra)]
        - vasp_spectra.energy[np.argmax(vasp_spectra.spectra)]
    )  # align to vasp max

    # mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(["vibrant", "no-latex"])
    plt.figure(figsize=(8, 5))
    plt.plot(vasp_spectra.energy, vasp_spectra.spectra, label="vasp", c="green")
    # plt.plot(feff_spectra.energy, feff_spectra.spectra , label="feff")
    plt.plot(
        feff_spectra.energy,
        feff_spectra.spectra / (0.529177**2),
        label="feff",
    )
    plt.plot(xs_energy, xs_spectra, label="xs", color="orange")
    plt.title(f"XAS of {id}")
    plt.legend()
    # plt.savefig(f"alignment_test_{id}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
