import scienceplots
from src.data.feff_data_raw import RAWDataFEFF
from src.data.feff_data import FEFFData
from src.data.vasp_data_raw import RAWDataVASP
from src.data.vasp_data import VASPData
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


if __name__ == "__main__":
    compound = "Cu"
    feff_raw_data = RAWDataFEFF(compound=compound)
    vasp_raw_data = RAWDataVASP(compound=compound)

    # ---------------------------
    # VASP data
    # ---------------------------
    id = ("mp-361", "000_Cu")
    vasp_gamma = 3  # <<- different from others
    vasp_data = VASPData(compound, vasp_raw_data.parameters[id], id, do_transform=False)
    vasp_data.reset().truncate().scale().broaden(gamma=vasp_gamma).align()

    # ---------------------------
    # FEFF data
    # ---------------------------
    feff_data = FEFFData(compound, feff_raw_data.parameters[id], id, do_transform=False)
    feff_data.reset().transform(include_emperical_truncation=False)
    feff_data.align(vasp_data)

    # ---------------------------
    # Experimental data
    # ---------------------------
    # extracted from paper
    exp_raw = np.loadtxt("dataset/cu2o_experiment/cu2o.csv", delimiter=",")
    exp_energy, exp_spectra = exp_raw[:, 0], exp_raw[:, 1]
    exp_data = deepcopy(feff_data)
    exp_data.energy = exp_energy
    exp_data.spectra = exp_spectra
    # exp uses different scale
    exp_data.spectra = exp_data.spectra - exp_data.spectra.min()
    exp_scale = vasp_data.spectra.max() / exp_data.spectra.max()
    exp_data.spectra = exp_scale * exp_data.spectra

    # -----------------------------
    # Plot all
    # -----------------------------
    plt.style.use(["default", "science"])
    plt.figure(figsize=(8, 5))
    kwargs = {"marker": "o", "markersize": 1, "linewidth": 1, "fillstyle": "none"}
    plt.plot(feff_data.energy, feff_data.spectra, label="FEFF", **kwargs)
    plt.plot(vasp_data.energy, vasp_data.spectra, label="VASP", **kwargs)
    plt.plot(exp_data.energy, exp_data.spectra, label="Experiment", **kwargs)
    plt.xlim(None, 9050)
    plt.xlabel("Energy (eV)", fontsize=18)
    plt.title(f"{id} \n vasp_gamma={vasp_gamma:.2f}", fontsize=18)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(f"experimental_alignment_{id}.pdf", dpi=300, bbox_inches="tight")
