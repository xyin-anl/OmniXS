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


compound_colors = [
    "#006BA4",
    "#FF800E",
    "#ABABAB",
    "#595959",
    "#5F9ED1",
    "#C85200",
    "#898989",
    "#A2C8EC",
    "#FFBC79",
    "#CFCFCF",
]

cmap = "tab10"
compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))


ASPECT_RATIO = 4 / 3
HEIGHT = 10
WEIGHT = HEIGHT / ASPECT_RATIO
DPI = 300
FONTSIZE = 14
COLOR = compound_colors[0]
plt.style.use(["default", "science", "tableau-colorblind10"])
mpl.rcParams["font.size"] = FONTSIZE
mpl.rcParams["axes.labelsize"] = FONTSIZE
mpl.rcParams["xtick.labelsize"] = FONTSIZE
mpl.rcParams["ytick.labelsize"] = FONTSIZE
mpl.rcParams["legend.fontsize"] = FONTSIZE
mpl.rcParams["figure.dpi"] = DPI
mpl.rcParams["figure.figsize"] = (WEIGHT, HEIGHT)
mpl.rcParams["savefig.dpi"] = DPI
mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"

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

    ax.patch.set_facecolor(compound_colors[i])
    ax.patch.set_alpha(0.2)
    bin_width = 0.01
    ax.hist(
        np.sqrt(model.mse_per_spectra),
        density=True,
        bins=np.arange(0, max(np.sqrt(model.mse_per_spectra)) + bin_width, bin_width),
        color=compound_colors[i],
        edgecolor="black",
        zorder=2,
        alpha=0.5,
    )
    kde = gaussian_kde(np.sqrt(model.mse_per_spectra))
    ax.plot(
        np.linspace(0, max(np.sqrt(model.mse_per_spectra)), 100),
        kde(np.linspace(0, max(np.sqrt(model.mse_per_spectra)), 100)),
        color="black",
        # color=tableau_colorblind10[i],
        linestyle="--",
        zorder=3,
    )
    x_max = max(x_max, np.quantile(np.sqrt(model.mse_per_spectra), 0.98))
    ax.text(
        0.9,
        0.9,
        (
            f"{compound}"
            if simulation_type == "FEFF"
            else f"{compound}_{simulation_type.lower()}"
        ),
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=FONTSIZE * 1.2,
        color="black",
    )
    # break

ax.set_xlim(x_min, 0.28)  # based on observation
for ax in axs[:, 0]:
    ax.set_ylabel("Density", fontsize=FONTSIZE * 1.2)
for ax in axs[-1, :]:
    ax.set_xlabel("RMSE", fontsize=FONTSIZE * 1.2)
axs[0, 0].legend(["KDE"], fontsize=FONTSIZE * 1.2, loc="upper left")
plt.tight_layout()
plt.savefig("rmse_hist.pdf", bbox_inches="tight", dpi=DPI)
plt.savefig("rmse_hist.png", bbox_inches="tight", dpi=DPI)

# %%
