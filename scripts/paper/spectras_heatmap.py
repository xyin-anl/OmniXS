# %%
from typing import Literal, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.colors import LogNorm, NoNorm, PowerNorm

from config.defaults import cfg
from scripts.plots.plot_all_spectras import MLDATAPlotter
from src.data.ml_data import DataQuery, load_xas_ml_data


# %%


all_data = {
    c: load_xas_ml_data(DataQuery(c, "FEFF"), filter_spectra_anomalies=True)
    for c in cfg.compounds
}
all_data["Ti_vasp"] = load_xas_ml_data(
    DataQuery("Ti", "VASP"), filter_spectra_anomalies=True
)
all_data["Cu_vasp"] = load_xas_ml_data(
    DataQuery("Cu", "VASP"), filter_spectra_anomalies=True
)


# %%


cmap = "tab10"
compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))
compound_colors = {
    c: compound_colors[i] for i, c in enumerate(cfg.compounds + ["Ti_vasp", "Cu_vasp"])
}


import matplotlib as mpl

FONTSIZE = 14
DPI = 300

plt.style.use(["default", "science", "tableau-colorblind10"])
mpl.rcParams["font.size"] = FONTSIZE
mpl.rcParams["axes.labelsize"] = FONTSIZE
mpl.rcParams["xtick.labelsize"] = FONTSIZE
mpl.rcParams["ytick.labelsize"] = FONTSIZE
mpl.rcParams["legend.fontsize"] = FONTSIZE
mpl.rcParams["figure.dpi"] = DPI
mpl.rcParams["savefig.dpi"] = DPI
mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"


def heatmap_of_lines(
    data: np.ndarray,
    ax=None,
    height: Union[Literal["same"], int] = "same",
    cmap="jet",
    # norm: matplotlib.colors.Normalize = LogNorm(),
    norm: matplotlib.colors.Normalize = LogNorm(
        vmin=1, vmax=100
    ),  # Adjust normalization
    aspect=0.618,  # golden ratio
    x_ticks=None,
    y_ticks=None,
    smooth=False,
):
    """
    Generate a heatmap from multiple lines of data.
    """

    if ax is None:
        ax = plt.gca()

    if smooth:

        from scipy.interpolate import interp1d

        x = np.arange(data.shape[1])
        x_new = np.linspace(0, data.shape[1] - 1, 1000)
        f_x = interp1d(x, data)
        data = f_x(x_new)

    # initialize heatmap to zeros
    width = data.shape[1]
    height = width if height == "same" else height

    heatmap = np.zeros((width, height))
    max_val = data.max()
    max_val *= 1.1  # add some padding
    for l in data:
        for x_idx, y_val in enumerate(l):
            y_idx = y_val / max_val * height
            y_idx = y_idx.astype(int)
            y_idx = np.clip(y_idx, 0, height - 1)
            heatmap[y_idx, x_idx] += 1

    colorbar = ax.imshow(
        heatmap,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect=aspect,
        interpolation="nearest",
    )

    if x_ticks is not None:
        x_ticks_pos = np.linspace(0, width, len(x_ticks))
        colorbar.axes.xaxis.set_ticks(x_ticks_pos, x_ticks)
    if y_ticks is not None:
        y_ticks_pos = np.linspace(0, height, len(y_ticks))
        colorbar.axes.yaxis.set_ticks(y_ticks_pos, y_ticks)

    return colorbar


plt.style.use(["default", "science"])

fig = plt.figure(figsize=(2 * 4, 2 * len(all_data) // 2))
# grid = plt.GridSpec(len(cfg.compounds) // 2, 2, hspace=0.0, wspace=0.0)
grid = plt.GridSpec(len(all_data) // 2, 2, hspace=0.0, wspace=0.0)
axs = [fig.add_subplot(grid[i, j]) for i in range(len(all_data) // 2) for j in range(2)]


# for c, ax, data in zip(cfg.compounds, axs, all_data):
print(all_data.keys())
for ax, (c, data) in zip(axs, all_data.items()):
    print(f"Plotting {c}...")
    # if c != "Mn":
    #     continue
    spectras = np.concatenate([data.train.y, data.val.y, data.test.y])
    heatmap_of_lines(
        spectras,
        ax=ax,
        aspect="auto",
        smooth=True,
        # smooth=False, # for dbugging
        cmap="jet",
    )
    ax.patch.set_facecolor(compound_colors[c])
    ax.patch.set_alpha(0.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(
        0.05,
        0.95,
        c,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="top",
        horizontalalignment="left",
    )
    # break

plt.tick_params(axis="x", which="both", bottom=False, top=False)
for i in [8, 9]:
    axs[i].set_xlabel("Index", fontsize=14)

for i in [0, 2, 4, 6, 8]:
    axs[i].set_ylabel("Target", fontsize=14)

fig.savefig(
    "feff_spetra_heatmap.pdf",
    bbox_inches="tight",
    dpi=300,
)

fig.savefig(
    "feff_spetra_heatmap.png",
    bbox_inches="tight",
    dpi=300,
)

# fig.show()

# %%
