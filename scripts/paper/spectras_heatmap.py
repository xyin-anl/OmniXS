# %%
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from config.defaults import cfg
from scripts.plots.plot_all_spectras import MLDATAPlotter
from src.data.ml_data import DataQuery, load_xas_ml_data
from utils.src.plots.heatmap_of_lines import heatmap_of_lines

# %%


all_data = [
    load_xas_ml_data(DataQuery(c, "FEFF"), filter_spectra_anomalies=True)
    for c in cfg.compounds
]

# %%


plt.style.use(["default", "science"])

fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(len(cfg.compounds) // 2, 2, hspace=0.0, wspace=0.0)
axs = [
    fig.add_subplot(grid[i, j])
    for i in range(len(cfg.compounds) // 2)
    for j in range(2)
]


for c, ax, data in zip(cfg.compounds, axs, all_data):
    print(f"Plotting {c}...")
    spectras = np.concatenate([data.train.y, data.val.y, data.test.y])
    heatmap_of_lines(
        spectras,
        ax=ax,
        aspect="auto",
    )
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

plt.tick_params(axis="x", which="both", bottom=False, top=False)
for i in [6, 7]:
    axs[i].set_xlabel("Index", fontsize=10)

for i in [0, 2, 4, 6]:
    axs[i].set_ylabel("Target", fontsize=10)

fig.savefig(
    "feff_spetra_heatmap.pdf",
    bbox_inches="tight",
    dpi=300,
)
fig.show()

# %%
