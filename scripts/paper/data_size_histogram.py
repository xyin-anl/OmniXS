# %%
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
import numpy as np
from config.defaults import cfg
from scripts.plots.plot_all_spectras import MLDATAPlotter
import matplotlib.pyplot as plt
from src.data.ml_data import load_all_data, load_xas_ml_data, DataQuery
from src.data.vasp_data_raw import RAWDataVASP
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data import VASPData

# %%

compound = "Cu"
simulation_type = "FEFF"

feff_raw = RAWDataFEFF(compound, simulation_type)
# %%

unconverged = [
    (mat_id, site)
    for mat_id in feff_raw._material_ids
    for site in feff_raw._sites.get(mat_id, [])
    if not feff_raw._check_convergence(mat_id, site)
]
print(f"Unconverged: {len(unconverged)}")
print(f"Total: {len(feff_raw)}")
ml_data = load_xas_ml_data(
    DataQuery(compound, simulation_type), filter_spectra_anomalies=False
)
print(f"ML Data: {len(ml_data.train.X) + len(ml_data.val.X) + len(ml_data.test.X)}")

# %%

feff_raw._material_ids
feff_raw._sites

# c = "Cu"
# load_xas_ml_data(DataQuery(c, "FEFF"), filter_spectra_anomalies=True)

# %%


all_data = [
    load_xas_ml_data(DataQuery(c, "FEFF"), filter_spectra_anomalies=True)
    for c in cfg.compounds
]

all_data = [np.concatenate([d.train.y, d.val.y, d.test.y]) for d in all_data]

# %%

nice_colors = [
    "#4e79a7",
    "#76b7b2",
    "#ff9da7",
    "#9b59b6",
    "#3498db",
    "#95a5a6",
    "#e74c3c",
    "#b07aa1",
]
import scienceplots

plt.style.use(["default", "science"])
plt.style.use(["default", "science", "grid"])
fontsize = 20
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
data_size = [len(x) for x in all_data]
ax.bar(cfg.compounds, data_size, color="#9b59b6")
ax.set_xlabel("Compounds", fontsize=fontsize)
ax.set_xticklabels(cfg.compounds, fontsize=fontsize / 1.5)
# plt.xticks([])
# put y ticks labels in multiple of 100
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_ylabel("Data Size", fontsize=fontsize, labelpad=0)
# name the compounds by puttin the text in middle of the bars

from matplotlib.ticker import FuncFormatter

ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000):,}"))
ax.text(
    0,
    1.02,
    r"$\times 10^3$",
    transform=ax.transAxes,
    fontsize=12,
    ha="left",
)

# make grid less visible
ax.grid(alpha=0.2)

for i, v in enumerate(data_size):
    ax.text(
        i,
        v + 0.005 * v,
        # put commas in the number
        f"{v:,}",
        ha="center",
        va="bottom",
        fontsize=12,
        # color="white",
    )

plt.tight_layout()
plt.savefig("data_size.png", dpi=300, bbox_inches="tight")

# %%
