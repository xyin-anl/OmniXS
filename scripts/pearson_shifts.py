import scienceplots
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from src.data.feff_data_transformations import FEFFDataModifier
from src.data.raw_data_feff import RAWDataFEFF

from src.data.raw_data_vasp import RAWDataVASP
from src.data.vasp_data_transformations import VASPDataModifier
from utils.src.plots.highlight_tick import highlight_tick


def compute_pearson_energy_shifts(compound="Ti", sample_size=50):
    feff_raw_data = RAWDataFEFF(compound=compound)
    vasp_raw_data = RAWDataVASP(compound=compound)

    common_ids = set(feff_raw_data.parameters.keys()).intersection(
        set(vasp_raw_data.parameters.keys())
    )
    ids = random.choices(list(common_ids), k=sample_size)
    pearson_alignments = []
    for id in ids:
        print(id)
        feff_data = FEFFDataModifier(feff_raw_data.parameters[id])
        vasp_data = VASPDataModifier(vasp_raw_data.parameters[id])
        shift = feff_data.spearman_align(vasp_data)
        pearson_alignments.append(shift)
    pearson_alignments = np.array(pearson_alignments)
    return pearson_alignments


if __name__ == "__main__":
    compound = "Ti"
    sample_size = 50
    pearson_alignments = compute_pearson_energy_shifts(compound, sample_size)

    # pearson_alignments = np.load("results/pearson_shifts.npy")
    plt.style.use(["default", "high-vis", "science"])
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(8, 6))
    plt.hist(
        pearson_alignments,
        bins=25,
        facecolor="tab:blue",
        edgecolor="k",
        label="pearson_shifts",
    )
    title = f"Energy alignment pearson shifts for {compound}\n"
    title += f"sample size : {sample_size}"
    plt.title(title)
    # use median
    mean = np.mean(pearson_alignments)
    highlight_tick(ax=plt.gca(), highlight=mean, axis="x")
    plt.axvline(
        mean,
        linestyle="--",
        label="mean_shift",
        color="red",
        linewidth=2,
    )
    plt.xlabel("Energy shift (eV)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"pearson_shifts_{compound}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
