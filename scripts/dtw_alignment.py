import random
import time

import matplotlib.pyplot as plt
import scienceplots

from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data import VASPData
from src.data.vasp_data_raw import RAWDataVASP
from utils.src.misc import icecream

if __name__ == "__main__":
    compound = "Ti"

    feff_raw_data = RAWDataFEFF(compound=compound)
    vasp_raw_data = RAWDataVASP(compound=compound)

    # seed = 42
    # random.seed(seed)
    sample_size = 10
    feff_ids = set(feff_raw_data.parameters.keys())
    vasp_ids = set(vasp_raw_data.parameters.keys())
    common_ids = feff_ids.intersection(vasp_ids)

    plt.style.use(["default", "science"])
    ids = random.choices(list(common_ids), k=sample_size)
    fig, axs = plt.subplots(len(ids), 1, figsize=(6, 3 * len(ids)))
    time_corr = []
    time_dtw = []
    for simulation_type in ["VASP", "FEFF"]:
        raw_data = vasp_raw_data if simulation_type == "VASP" else feff_raw_data
        data_class = VASPData if simulation_type == "VASP" else FEFFData
        for ax, id in zip(axs, ids):
            data = data_class(
                compound=compound,
                params=raw_data.parameters[id],
            )
            data.reset().transform(include_emperical_truncation=False)
            if data.simulation_type == "FEFF":
                t1 = time.time()
                data.align(VASPData(compound, vasp_raw_data.parameters[id]))
                del_t = time.time() - t1
                time_corr.append(del_t)
                ic("time_taken_for_corr", del_t)
            data.reset().transform(include_emperical_truncation=True)
            ax.plot(
                data.energy,
                data.spectra,
                label=f"{data.simulation_type}_{id}",
                linestyle="-",
            )

            # doing again for dtw based
            if data.simulation_type == "FEFF":
                data = data_class(
                    compound=compound,
                    params=raw_data.parameters[id],
                )
                data.reset().transform(include_emperical_truncation=False)
                t1 = time.time()
                shift = data_class.dtw_shift(
                    data,
                    VASPData(compound, vasp_raw_data.parameters[id]),
                )
                del_t = time.time() - t1
                time_dtw.append(del_t)
                ic("time_taken_for_dtw", del_t)
                data.align_energy(-shift)
                data.reset().transform()  # includes truncation
                ax.plot(
                    data.energy,
                    data.spectra,
                    label=f"{data.simulation_type}_{id}_dtw",
                    linestyle="--",
                )

            ax.legend()
            ax.sharex(axs[0])

    axs[-1].set_xlabel("Energy (eV)", fontsize=18)

    # save each axis of figure
    for i, ax in enumerate(axs):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"dtw_vs_corr_alignment_{compound}_{i}.pdf", bbox_inches=extent)

    plt.suptitle(f"Per-spectra alignment samples: {compound}", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"dtw_vs_corr_alignment_{compound}.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    import numpy as np

    plt.clf()
    plt.hist(
        np.array(time_corr) / np.array(time_dtw),
        label="time_taken_corr/time_taken_dtw",
        color="tab:green",
        edgecolor="green",
        alpha=0.5,
        linewidth=3,
    )
    plt.title(
        "Speed  of alignment algorithms \n  Correlation_based vs DTW_based ",
        fontsize=20,
    )
    plt.ylabel("count", fontsize=20)
    plt.xlabel("time_taken corr_based/dtw_based", fontsize=20)
    plt.savefig("time_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.show()
