# %%
from p_tqdm import p_map
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
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
from src.data.ml_data import FeatureProcessor
from src.models.trained_models import Trained_FCModel, MeanModel, LinReg


# %%


plt.style.use(["default", "science", "no-latex"])
DPI = 300
FONTSIZE = 14
plt.style.use(["default", "science", "tableau-colorblind10", "no-latex"])
mpl.rcParams["font.size"] = FONTSIZE
mpl.rcParams["figure.dpi"] = DPI
mpl.rcParams["savefig.dpi"] = DPI
mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"

# =============================================================================
# PLOT PCA EXPLAINED VARIANCE for all compounds for ACSF and SOAP
# =============================================================================
# for simulation_type in ["ACSF", "SOAP", "FEFF"]:

# fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=DPI, sharey=True)
fig = plt.figure(figsize=(6, 10), dpi=DPI)
plt.style.use(["default", "science", "no-latex"])  # TODO: fix latex
# fig.subplots_adjust(wspace=0, hspace=0)

gs = gridspec.GridSpec(3, 1, wspace=0, hspace=0)
axs = [plt.subplot(gs[i]) for i in range(3)]

cmap = "tab10"
compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))

for ax, simulation_type in zip(axs, ["ACSF", "SOAP", "FEFF"]):
    pcas = p_map(
        lambda c: FeatureProcessor(
            DataQuery(c, simulation_type),
            data_splits=load_xas_ml_data(DataQuery(c, simulation_type)),
        ).pca,
        cfg.compounds,
    )
    pcas = {c: v for c, v in zip(cfg.compounds, pcas)}
    pca_dims = {c: pca.n_components_ for c, pca in pcas.items()}

    for i, c in enumerate(np.array(cfg.compounds)[np.argsort(list(pca_dims.values()))]):

        # pca = feature_processors[c].pca
        pca = pcas[c]
        n_components = pca.n_components_

        label = r"{c}: {pca.n_components_}"
        kwargs = {"label": label, "marker": "o", "markersize": 2}

        x = np.arange(1, n_components + 1)
        y = pca.explained_variance_ratio_
        y = np.cumsum(y)
        ax.plot(x, y, color=compound_colors[i], **kwargs)
    pre_reduced_dims = [
        load_xas_ml_data(
            DataQuery(c, simulation_type),
            pca_with_std_scaling=False,
        ).train.X.shape[1]
        for c in cfg.compounds
    ]
    assert len(set(pre_reduced_dims)) == 1
    # misc

    x_value = cfg.dscribe.pca.n_components

    # ax.set_title(rf"{simulation_type}" if simulation_type != "FEFF" else r"M3GNet")

    txt = rf"{simulation_type}" if simulation_type != "FEFF" else r"M3GNet"
    txt += f"\nDimension: {pre_reduced_dims[0]}"
    ax.text(
        0.5,
        0.5,
        txt,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=FONTSIZE * 1.2,
        color="black",
    )

    # draw horizontal line at y = x_value
    ax.axhline(y=x_value, color="gray", linestyle="--")
    ax.set_yticks(np.arange(0.2, 1.05, 0.2))
    ax.set_yticklabels([f"{i:.1f}" for i in np.arange(0.2, 1.05, 0.2)])
    ax.set_ylabel("η(d)", fontsize=FONTSIZE)
    ax.set_ylim(0.1, 1.05)

    ax.set_xlim(0, 60)


# use latex

axs[-1].set_xlabel(
    "Principal Components (d)",
    fontsize=FONTSIZE,
)
axs[-1].set_xticks(np.arange(0, 61, 10))
axs[-1].set_xticklabels([f"{i}" for i in np.arange(0, 61, 10)])

handles = [
    mlines.Line2D(
        [],
        [],
        color=compound_colors[i],
        marker="o",
        # linestyle="-",
        markersize=10,
        label=c,
    )
    for i, c in enumerate(cfg.compounds)
]

# add handle for horizontal line
handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        linestyle="--",
        label=f"{x_value}",
    ),
    *handles,
]
# put legend on last axis
axs[0].legend(
    handles=handles,
    title=r"Compound",
    fontsize=FONTSIZE * 0.8,
    title_fontsize=FONTSIZE * 0.8,
    # to right most of 3rd subplot
    loc="center right",
    # bbox_to_anchor=(1.08, 0.5),
)

# fig.legend(
#     handles=handles,
#     title=r"Compound",
#     fontsize=FONTSIZE * 0.8,
#     title_fontsize=FONTSIZE * 0.8,
#     # to right most of 3rd subplot
#     loc="center right",
#     bbox_to_anchor=(1.08, 0.5),
# )

fig.tight_layout()
fig.savefig("pca_explained_variance.pdf", bbox_inches="tight", dpi=DPI)

# %%
