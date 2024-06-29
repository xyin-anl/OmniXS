# %%
import matplotlib as mpl
from src.models.trained_models import MeanModel
from scipy.stats import gaussian_kde
from config.defaults import cfg
import scienceplots
from matplotlib import pyplot as plt
import random
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from scripts.plots.plot_all_spectras import MLDATAPlotter
from src.data.ml_data import load_all_data
from src.models.trained_models import Trained_FCModel
from src.data.ml_data import DataQuery, load_xas_ml_data
import numpy as np


# %%


cmap = "tab10"
compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))


ASPECT_RATIO = 4 / 3
HEIGHT = 10
WEIGHT = HEIGHT / ASPECT_RATIO
DPI = 300
FONTSIZE = 18
COLOR = compound_colors[0]
plt.style.use(["default", "science", "tableau-colorblind10"])

# mpl.rcParams["font.size"] = FONTSIZE
# mpl.rcParams["axes.labelsize"] = FONTSIZE
# mpl.rcParams["xtick.labelsize"] = FONTSIZE
# mpl.rcParams["ytick.labelsize"] = FONTSIZE
# mpl.rcParams["legend.fontsize"] = FONTSIZE
# mpl.rcParams["figure.dpi"] = DPI
# mpl.rcParams["figure.figsize"] = (WEIGHT, HEIGHT)
# mpl.rcParams["savefig.dpi"] = DPI
# mpl.rcParams["savefig.format"] = "pdf"
# mpl.rcParams["savefig.bbox"] = "tight"

model_name = "ft_tl"
simulation_types = ["FEFF"] * len(cfg.compounds) + ["VASP", "VASP"]
compounds = cfg.compounds + ["Ti", "Cu"]


COLS = 2
ROWS = len(compounds) // COLS

HEIGHT = 10
WEIGHT = HEIGHT / ASPECT_RATIO

fig, axs = plt.subplots(
    ROWS,
    COLS,
    figsize=(WEIGHT, HEIGHT),
    dpi=DPI,
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0, "wspace": 0},
)


x_min = 0
x_max = 0
UPPER_QUANTILE = 0.95
for i, ax, (compound, simulation_type) in zip(
    range(len(compounds)), axs.flatten(), zip(compounds, simulation_types)
):

    model = Trained_FCModel(
        DataQuery(compound, simulation_type),
        name="universal_tl" if compound == "ALL" else model_name,
    )

    # ax.patch.set_facecolor(compound_colors[i])
    ax.patch.set_alpha(0.2)
    bin_width = 0.01
    bin_count = 30
    ax.hist(
        # np.sqrt(model.mse_per_spectra),
        np.log10(model.mse_per_spectra),
        density=True,
        bins=np.linspace(-4, 0, bin_count),
        color=compound_colors[i],
        edgecolor="black",
        zorder=1,
        alpha=0.5,
    )

    # # kde = gaussian_kde(np.sqrt(model.mse_per_spectra))
    # kde = gaussian_kde(model.mse_per_spectra)
    # ax.plot(
    #     np.linspace(0, max(model.mse_per_spectra), 100),
    #     kde(np.linspace(0, max(model.mse_per_spectra), 100)),
    #     # np.linspace(0, max(np.sqrt(model.mse_per_spectra)), 100),
    #     # kde(np.linspace(0, max(np.sqrt(model.mse_per_spectra)), 100)),
    #     color="black",
    #     # color=tableau_colorblind10[i],
    #     linestyle="--",
    #     zorder=3,
    # )

    # x_max = max(x_max, np.quantile(np.sqrt(model.mse_per_spectra), 0.98))
    # x_max = max(x_max, np.quantile(model.mse_per_spectra, 0.98))
    ax.text(
        0.9,
        0.9,
        (
            f"{compound}"
            # if simulation_type == "FEFF"
            # else f"{compound}_{simulation_type.lower()}"
        ),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
        color="black",
        fontweight="bold",
        bbox=dict(
            facecolor=compound_colors[i],
            alpha=0.5,
            edgecolor=compound_colors[i],
        ),
    )

    xticks = np.arange(-4, 1, 1)
    ax.set_xticks(xticks[:-1])
    ax.set_xticklabels(
        [r"$10^{" + f"{x}" + "}$" for x in xticks[:-1]], fontsize=FONTSIZE * 0.8
    )
    ax.set_xlim(xticks[0], xticks[-1])

    # yticks = np.arange(0.2, 1.5, 0.3)
    # yticks = [0.4, 0.8, 1.2]
    yticks = [0.4, 0.8, 1.2]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=FONTSIZE * 0.8)
    ax.minorticks_off()

    # add text right below vasp legend
    if simulation_type == "VASP":
        ax.text(
            0.86,
            0.7,
            "VASP",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=FONTSIZE * 0.5,
            color="black",
        )

    # add vertical line where there is MSE of the MeanModel
    mean_model_mse = MeanModel(DataQuery(compound, simulation_type)).mse
    ax.axvline(
        np.log10(mean_model_mse),
        color="black",
        linestyle=":",
        alpha=0.5,
        label=r"$\text{MSE}_{\text{baseline}}$",
        # put it behind histogram
        zorder=0,
    )


# for ax in axs[:, 0]:
#     ax.set_ylabel("Density", fontsize=FONTSIZE)
axs[2, 0].set_ylabel("Density", fontsize=FONTSIZE)

for ax in axs[-1, :]:
    ax.set_xlabel("MSE", fontsize=FONTSIZE)

# add legend about mean model mse in first subplot
axs[0, 0].legend(fontsize=FONTSIZE * 0.8)


# axs[0, 0].legend(["KDE"], fontsize=FONTSIZE, loc="upper left")
plt.tight_layout()
plt.savefig("mse_hist.pdf", bbox_inches="tight", dpi=DPI)
plt.savefig("mse_hist.png", bbox_inches="tight", dpi=DPI)

# %%
