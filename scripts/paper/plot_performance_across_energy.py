# %%
from typing import Literal
import warnings
from config.defaults import cfg
from dataclasses import dataclass
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch

from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import MeanModel, Trained_FCModel


def winrates(
    compound,
    simulation_type,
):

    diff = (
        Trained_FCModel(
            DataQuery(compound, simulation_type), name="per_compound_tl"
        ).absolute_errors
        - Trained_FCModel(
            DataQuery(compound, simulation_type), name="ft_tl"
        ).absolute_errors
    )

    def rate(axis):
        return np.mean(np.mean(diff, axis=axis) > 0) * 100

    # return rate(0), rate(1)

    out = {}
    out["energy"] = rate(0)
    out["compound"] = rate(1)
    return out


from config.defaults import cfg

wins = {(c, "FEFF"): winrates(c, "FEFF") for c in cfg.compounds}
wins["Cu", "VASP"] = winrates("Cu", "VASP")
wins["Ti", "VASP"] = winrates("Ti", "VASP")

# round to two decimal
for k, v in wins.items():
    for kk, vv in v.items():
        wins[k][kk] = round(vv, 2)



# %%


def compare_mse_per_energy(compounds, simulation_type="FEFF", axs=None, fontsize=18):
    if axs is None:
        fig, axs = plt.subplots(
            len(compounds),
            1,
            figsize=(9, 12),
            sharex=True,
        )
    axs = axs.flatten()

    LINEWIDTH = 0.4
    compound_colors = {c: plt.cm.tab10.colors[i] for i, c in enumerate(compounds)}
    for idx, compound in enumerate(compounds):

        from src.models.trained_models import Trained_FCModel

        axs[idx].text(
            0.13,
            0.85,
            r"$w_{\text{e}}=$"
            + f" {winrates(compound, simulation_type)['energy']:.0f}\%",
            fontsize=18,
            transform=axs[idx].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            color=plt.cm.tab10.colors[idx],
            fontweight="bold",
        )

        expertXAS_residues = Trained_FCModel(
            query=DataQuery(compound=compound, simulation_type=simulation_type),
            name="per_compound_tl",
        ).absolute_errors

        tunedXAS_residues = Trained_FCModel(
            query=DataQuery(compound=compound, simulation_type=simulation_type),
            name="ft_tl",
        ).absolute_errors

        energy_points = 0.25 * np.arange(len(expertXAS_residues[0]))

        differences = expertXAS_residues - tunedXAS_residues

        avg_residues = np.mean(differences, axis=0)
        baseline = np.mean(
            MeanModel(DataQuery(compound, simulation_type)).predictions[0]
        )
        avg_residues = avg_residues / baseline * 100

        # bar plot of avg_residues and color them based on sign
        axs[idx].bar(
            energy_points,
            abs(avg_residues),
            color=np.where(avg_residues > 0, "green", "red"),
            width=0.25,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.5,
        )

        # improvements = np.clip(differences, a_min=0, a_max=None)
        # regressions = np.clip(differences, a_min=None, a_max=0)
        # assert np.all(improvements >= 0)
        # assert np.all(regressions <= 0)
        # baseline = np.mean(
        #     MeanModel(DataQuery(compound, simulation_type)).predictions[0]
        # )
        # improvements = improvements / baseline * 100
        # regressions = regressions / baseline * 100

        # # vlines from improvements to 0
        # axs[idx].vlines(
        #     energy_points,
        #     0,
        #     improvements,
        #     color="green",
        #     # alpha=0.5,
        # )

        # axs[idx].vlines(
        #     energy_points,
        #     0,
        #     regressions,
        #     color="red",
        #     # alpha=0.5,
        # )

        # # fill from improvements to regressions if improvements > 0
        # for i, (imp, reg) in enumerate(zip(improvements, regressions)):
        #     if imp > reg:
        #         axs[idx].fill_between(
        #             [energy_points[i] - 0.125, energy_points[i] + 0.125],
        #             imp - reg,
        #             0,
        #             color="green",
        #             alpha=0.4,
        #             zorder=1,
        #         )
        #     else:
        #         axs[idx].fill_between(
        #             [energy_points[i] - 0.125, energy_points[i] + 0.125],
        #             reg - imp,
        #             0,
        #             color="red",
        #             alpha=0.4,
        #         )

        # axs[idx].set_ylim(0, None)

        axs[idx].text(
            0.04,
            (
                0.85
                # if not (compound in ["Cu", "Mn"] and simulation_type == "FEFF")
                # else 0.25
            ),
            compound,
            fontsize=18,
            transform=axs[idx].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor=compound_colors[compound],
                alpha=0.4,
                edgecolor="white",
            ),
        )

        if simulation_type == "VASP":
            # add text bleow the compound name saying VASP with size 12
            axs[idx].text(
                0.03,
                0.6,
                "VASP",
                fontsize=12,
                # color="#101010",
                color=plt.cm.tab10.colors[idx],
                transform=axs[idx].transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.2, edgecolor="white"),
            )

        # continue

        axs[idx].set_xlim(-0.25, max(energy_points) + 0.25)
        axs[idx].set_ylim(0, None)

        axs_mean_spectra = axs[idx].twinx()
        mean_spectra = MeanModel(DataQuery(compound, simulation_type)).predictions[0]

        mean_spectra = mean_spectra  # TODO: remove hardcoding
        axs_mean_spectra.plot(
            energy_points,
            mean_spectra,
            color="#101010",
            # color="black",
            # color=plt.cm.tab10.colors[idx],
            linewidth=LINEWIDTH * 1.5,
            linestyle="--",
            label="Mean spectra",
            zorder=0,
        )
        axs_mean_spectra.spines["right"].set_color("gray")

        # put ticks on 0 and 50 percent of the y axis that
        y_max = axs[idx].get_ylim()[1]
        y_tick = y_max / 2
        y_tick = round(y_tick * 4) / 4
        axs[idx].set_yticks([0, y_tick])

        axs[idx].yaxis.set_tick_params(labelsize=20, right=False)
        axs[idx].xaxis.set_tick_params(labelsize=20)
        axs[idx].tick_params(axis="both", which="minor", right=False)
        axs[idx].tick_params(axis="x", which="both", top=False)

        axs_mean_spectra.yaxis.set_tick_params(labelsize=20)
        axs_mean_spectra.xaxis.set_tick_params(labelsize=20)
        axs_mean_spectra.set_yticks([0, 1])
        axs_mean_spectra.set_ylim(0, None)


fig = plt.figure(figsize=(8, 18))
plt.style.use(["default", "science"])
gs = fig.add_gridspec(10, 1, hspace=0.035, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
FONTSIZE = 26

compare_mse_per_energy(cfg.compounds, axs=axs, fontsize=FONTSIZE)

# ADD VASP AS ELL
compare_mse_per_energy(
    ["Ti", "Cu"],
    axs=axs[-2:],
    simulation_type="VASP",
)

axs[0].legend(
    fontsize=FONTSIZE * 0.7,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.8),
    ncol=2,
    frameon=False,
)
axs[-1].set_xlabel(r"$\Delta E$ (eV)", fontsize=FONTSIZE)

fig.text(
    0.035,
    0.5,
    r"$|\tilde{I}(\Delta E)|$",
    va="center",
    rotation="vertical",
    fontsize=FONTSIZE * 1.1,
    ha="center",
)

# common y label at right side of figure for all twin axses that says r"<\mu>"
fig.text(
    0.96,
    0.5,
    # r"$\left<\mu\right>_E \times 10^3$",  # TODO: remove hardcoding
    r"$\mu_{\text{baseline}}$",
    va="center",
    rotation=-90,
    fontsize=FONTSIZE * 1.1,
    ha="center",
)

# add legend on top of figure with 2 cols for red/green bars
handles = [
    plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.5),
    plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.5),
]
labels = [r"$\tilde{I}(\Delta E) < 0$", r"$\tilde{I}(\Delta E) > 0$"]
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.9225),
    ncol=2,
    frameon=False,
    fontsize=FONTSIZE * 0.7,
)

fig.tight_layout()
fig.savefig("performance_across_energy.pdf", bbox_inches="tight", dpi=300)

# %%

# now partition based on spectra and plot vertical violin plots for each partition
