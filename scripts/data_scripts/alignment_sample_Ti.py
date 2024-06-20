import matplotlib.pyplot as plt
import numpy as np

from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data_raw import RAWDataVASP
from src.data.vasp_data import VASPData
import scienceplots

if __name__ == "__main__":
    compound = "Ti"
    id = ("mp-390", "000_Ti")
    vasp = RAWDataVASP(compound=compound)
    vasp_spectra = VASPData(compound=compound, params=vasp.parameters[id])
    # gamms 0.89 * 2 is only for the comparision with xs
    vasp_spectra.reset().truncate().scale().broaden(gamma=0.89 * 2).align()

    feff = RAWDataFEFF(compound=compound)
    feff_spectra = FEFFData(compound=compound, params=feff.parameters[id])
    feff_spectra.reset().transform()
    feff_spectra.align(vasp_spectra)

    xs_mp_390 = np.load("dataset/misc/xs-mp-390.npy")
    xs_energy, xs_spectra = xs_mp_390[0], xs_mp_390[1]
    xs_energy -= (
        xs_energy[np.argmax(xs_spectra)]
        - vasp_spectra._energy[np.argmax(vasp_spectra._spectra)]
    )  # align to vasp max

    plt.style.use(["vibrant"])
    plt.figure(figsize=(8, 5))
    plt.plot(vasp_spectra._energy, vasp_spectra._spectra, label="vasp", c="green")
    plt.plot(feff_spectra._energy, feff_spectra._spectra, label="feff")
    plt.plot(xs_energy, xs_spectra, label="xs", color="orange")
    plt.title(f"XAS of {id}")
    plt.legend()
    plt.savefig(f"alignment_sample_{id}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
