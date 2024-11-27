# %%
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from config.defaults import cfg

from omnixas.scripts.plots.constants import FEFFSplits, VASPSplits
from omnixas.utils.plots import plot_line_heatmap

# %%

FEFF_data = {k.value: v for k, v in FEFFSplits.items()}
VASP_data = {k.value + "_VASP": v for k, v in VASPSplits.items()}

# %%

PLOT = "FEFF"  # FEFF or VASP

FONTSIZE = 20
DPI = 300
INTERPOLATE = 1000  # or NONE or int (1000)
# INTERPOLATE = None  # or NONE or int (1000)

cmap = "tab10"
compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))
compound_colors = {
    c: compound_colors[i] for i, c in enumerate(cfg.compounds + ["Ti_VASP", "Cu_VASP"])
}

plt.style.use(["default", "science"])

all_data = VASP_data if PLOT == "VASP" else FEFF_data


COLS = 2 if PLOT == "VASP" else 4
ROWS = 1 if PLOT == "VASP" else 2

fig = plt.figure(figsize=(COLS * 4, ROWS * 3.5))
grid = plt.GridSpec(ROWS, COLS, hspace=0.015, wspace=0.02, figure=fig)
axs = [fig.add_subplot(grid[i, j]) for i in range(ROWS) for j in range(COLS)]
# axs = [fig.add_subplot(grid[i, j]) for i in range(len(all_data) // 2) for j in range(2)]

# # remove_idx = [2, 3]
# remove_idx = [0, 3]
# for i in remove_idx:
#     fig.delaxes(axs[i])
# axs = [ax for i, ax in enumerate(axs) if i not in remove_idx]

for ax, (c, data) in zip(axs, all_data.items()):
    logger.info(f"Plotting {c}...")
    spectras = np.concatenate([data.train.y, data.val.y, data.test.y])

    plot_line_heatmap(
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


range_values = range(len(axs)) if PLOT == "VASP" else range(4, len(axs))
# for i in range(4, len(axs)):  # FEFF
# for i in range(len(axs)):  # VASP
for i in range_values:
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


# single y label on left side that says $\mu(E) (a.u.)$
fig.text(
    0.078 if PLOT == "VASP" else 0.098,
    0.5,
    r"$\mu(E)$ (a.u.)",
    va="center",
    rotation="vertical",
    fontsize=FONTSIZE * 1.2,
)

fig.savefig(
    "spectra_heatmap_" + PLOT.lower() + ".pdf",
    bbox_inches="tight",
    dpi=300,
)


# %%
