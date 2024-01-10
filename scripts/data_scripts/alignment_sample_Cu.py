import random

import scienceplots
from matplotlib import pyplot as plt

from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data import VASPData
from src.data.vasp_data_raw import RAWDataVASP

if __name__ == "__main__":
    compound = "Cu"
    feff_raw_data = RAWDataFEFF(compound=compound)
    vasp_raw_data = RAWDataVASP(compound=compound)

    samples = 4
    seed = 42
    vasp_gamma = 1.19

    random.seed(seed)
    common_ids = set(feff_raw_data.parameters.keys()).intersection(
        vasp_raw_data.parameters.keys()
    )
    ids = random.sample(list(common_ids), samples)
    ids[0] = ("mp-361", "000_Cu")

    fig, axs = plt.subplots(samples, 1, figsize=(samples * 2, 3 * samples))
    for id, ax in zip(ids, axs):
        feff_data = FEFFData(
            compound,
            feff_raw_data.parameters[id],
            id,
            do_transform=False,
        )

        vasp_data = VASPData(
            compound,
            vasp_raw_data.parameters[id],
            id,
            do_transform=False,
        )

        feff_data.transform(include_emperical_truncation=False)
        vasp_data.reset().truncate().scale().broaden(gamma=vasp_gamma).align()
        feff_data, vasp_data = feff_data.resample(), vasp_data.resample()

        plt.style.use(["default", "science", "grid"])
        kwargs = {"markersize": 2, "linewidth": 1, "marker": "o"}
        ax.plot(vasp_data._energy, vasp_data._spectra, label="VASP", **kwargs)
        ax.plot(feff_data._energy, feff_data._spectra, label="FEFF", **kwargs)
        ax.set_xlabel("Energy (eV)")
        ax.set_title(f"ID: {id}")
        ax.set_xlim(None, 9040)
        ax.legend()

    plt.suptitle(
        f"Sample transfer learning for {compound} \n vasp_gamma = {vasp_gamma} "
    )
    plt.tight_layout()
    plt.savefig(
        f"sample_transfer_learning_{compound}_gamma_{vasp_gamma}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
