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


all_data = {}

all_data["Ti_VASP"] = load_xas_ml_data(
    DataQuery("Ti", "VASP"), filter_spectra_anomalies=True
)
all_data["Cu_VASP"] = load_xas_ml_data(
    DataQuery("Cu", "VASP"), filter_spectra_anomalies=True
)


for c in cfg.compounds:
    all_data[c] = load_xas_ml_data(DataQuery(c, "FEFF"), filter_spectra_anomalies=True)


# %%


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
    interpolate: Union[None, int] = None,
):
    """
    Generate a heatmap from multiple lines of data.
    """

    if ax is None:
        ax = plt.gca()

    if interpolate is not None:

        from scipy.interpolate import interp1d

        x = np.arange(data.shape[1])
        x_new = np.linspace(0, data.shape[1] - 1, interpolate)
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


FONTSIZE = 20
DPI = 300
INTERPOLATE = 1000  # or NONE or int (1000)

cmap = "tab10"
compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))
compound_colors = {
    c: compound_colors[i] for i, c in enumerate(cfg.compounds + ["Ti_VASP", "Cu_VASP"])
}

plt.style.use(["default", "science"])

# fig = plt.figure(figsize=(2 * 4, 2 * len(all_data) // 2))
# # grid = plt.GridSpec(len(cfg.compounds) // 2, 2, hspace=0.0, wspace=0.0)
# grid = plt.GridSpec(len(all_data) // 2, 2, hspace=0.0, wspace=0.0)
# axs = [fig.add_subplot(grid[i, j]) for i in range(len(all_data) // 2) for j in range(2)]

COLS = 4
ROWS = 3
fig = plt.figure(figsize=(COLS * 4, ROWS * 3.5))
grid = plt.GridSpec(ROWS, COLS, hspace=0.015, wspace=0.02, figure=fig)
axs = [fig.add_subplot(grid[i, j]) for i in range(ROWS) for j in range(COLS)]
# axs = [fig.add_subplot(grid[i, j]) for i in range(len(all_data) // 2) for j in range(2)]

# remove_idx = [2, 3]
remove_idx = [0, 3]
for i in remove_idx:
    fig.delaxes(axs[i])
axs = [ax for i, ax in enumerate(axs) if i not in remove_idx]

for ax, (c, data) in zip(axs, all_data.items()):
    print(f"Plotting {c}...")
    spectras = np.concatenate([data.train.y, data.val.y, data.test.y])

    heatmap_of_lines(
        spectras,
        ax=ax,
        aspect="auto",
        interpolate=INTERPOLATE,
        cmap="jet",
    )

    # ax.patch.set_facecolor(compound_colors[c]) # set background color
    # ax.patch.set_alpha(0.2)

    ax.text(
        0.05,
        0.95,
        c.split("_")[0],
        transform=ax.transAxes,
        fontsize=FONTSIZE,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor=compound_colors[c], alpha=0.5, edgecolor="white"),
    )
    if "VASP" in c:
        ax.text(
            0.02,
            0.75,
            "VASP",
            transform=ax.transAxes,
            fontsize=FONTSIZE * 0.8,
            verticalalignment="bottom",
            horizontalalignment="left",
            # bbox=dict(facecolor="white", alpha=0.5, edgecolor="white"),
        )

    ax.set_xticklabels([])
    ax.tick_params(axis="x", which="both", bottom=True, top=False)

    ax.set_yticks([])


# for i in range(4, len(axs)):
for i in range(len(axs)):
    axs[i].set_xlabel(r"$\Delta E$ (eV)", fontsize=FONTSIZE)
    xticks = np.array([20, 40, 60, 80, 100, 120]).astype(int)
    # xticks = np.array([10, 30, 50, 70, 90, 110, 130]).astype(int)
    xtick_labels = [int(x * 0.25) for x in xticks]

    # ====== Scaling ======
    if INTERPOLATE:
        xticks = ((xticks / 141) * 1000).astype(int)  # when scaling is done
        axs[i].set_xlim(0, 1000)  # when scaling is done
    # ============

    axs[i].set_xticks(xticks)
    axs[i].set_xticklabels(xtick_labels, fontsize=FONTSIZE * 0.8)


for i in [0, 2, 6]:
    axs[i].set_ylabel(r"$\mu(E)$ (a.u.)", fontsize=FONTSIZE)


# # single y label on left side that says $\mu(E) (a.u.)$
# fig.text(
#     0.098,
#     0.5,
#     r"$\mu(E)$ (a.u.)",
#     va="center",
#     rotation="vertical",
#     fontsize=FONTSIZE * 1.2,
# )

fig.savefig(
    "spectra_heatmap.pdf",
    bbox_inches="tight",
    dpi=300,
)

# %%
