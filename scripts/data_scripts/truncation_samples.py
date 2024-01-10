import random
from copy import deepcopy
import matplotlib.pyplot as plt
from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data import VASPData
from src.data.vasp_data_raw import RAWDataVASP
import scienceplots

if __name__ == "__main__":
    compound = "Ti"
    feff_raw_data = RAWDataFEFF(compound=compound)
    vasp_raw_data = RAWDataVASP(compound=compound)

    sample_size = 5
    seed = 42
    feff_ids = set(feff_raw_data.parameters.keys())
    vasp_ids = set(vasp_raw_data.parameters.keys())
    common_ids = feff_ids.intersection(vasp_ids)

    plt.style.use(["default", "science", "grid"])
    random.seed(seed)
    ids = random.choices(list(common_ids), k=sample_size)
    fig, axs = plt.subplots(len(ids), 1, figsize=(8, 2 * len(ids)))
    for simulation_type in ["VASP", "FEFF"]:
        raw_data = vasp_raw_data if simulation_type == "VASP" else feff_raw_data
        data_class = VASPData if simulation_type == "VASP" else FEFFData
        for ax, id in zip(axs, ids):
            data = data_class(
                compound=compound,
                params=raw_data.parameters[id],
            )

            data.reset().transform(include_emperical_truncation=False)

            ax.plot(
                data.energy,
                data.spectra,
                label=f"{data.simulation_type} full",
                linestyle="--",
                color="red",
            )

            data.reset().transform(include_emperical_truncation=True)
            ax.plot(
                data.energy,
                data.spectra,
                label=f"{data.simulation_type} chopped",
                linestyle="-",
                color="tab:blue",
            )

            ax.legend()
            ax.sharex(axs[0])
    axs[-1].set_xlabel("Energy (eV)", fontsize=18)
    plt.suptitle(f"VASP truncation samples: {compound}", fontsize=18)
    plt.tight_layout()
    plt.savefig(
        f"vasp_truncation_examples_{compound}.pdf", bbox_inches="tight", dpi=300
    )
    plt.show()
    #
