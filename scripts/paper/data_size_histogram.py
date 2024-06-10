# %%
import os
import pickle
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


def get_data_sizes(compound, simulation_type):

    def data_count(ml_splits):
        return len(ml_splits.train.X) + len(ml_splits.val.X) + len(ml_splits.test.X)

    pre_filter_size = data_count(
        load_xas_ml_data(
            DataQuery(compound, simulation_type), filter_spectra_anomalies=False
        )
    )
    post_filter_size = data_count(
        load_xas_ml_data(
            DataQuery(compound, simulation_type), filter_spectra_anomalies=True
        )
    )
    anamolies = pre_filter_size - post_filter_size

    raw_data = (
        RAWDataFEFF(compound, simulation_type)
        if simulation_type == "FEFF"
        else RAWDataVASP(compound)
    )
    if simulation_type == "VASP":
        unconverged_count = 0
    else:
        unconverged_count = len(
            [
                (mat_id, site)
                for mat_id in raw_data._material_ids
                for site in raw_data._sites.get(mat_id, [])
                if not raw_data._check_convergence(mat_id, site)
            ]
        )

    print(f"Compound: {compound}, Simulation Type: {simulation_type}")
    print(f"RAW Data Size: {raw_data.total_sites}")
    print(f"Unconverged: {unconverged_count}")
    print(f"Missing: {len(raw_data.missing_data)}")
    print(f"Pre-Filter Size: {pre_filter_size}")
    print(f"Anomalies: {anamolies}")
    print(f"Post-Filter Size: {post_filter_size}")

    return {
        "total": len(raw_data),
        "unconverged": unconverged_count,
        "anomalies": anamolies,
        "ml": post_filter_size,
    }


# %%


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def darken_color(color, factor=0.8):
    if isinstance(color, str):
        color = hex_to_rgb(color)
    return tuple(max(0, min(1, c * factor)) for c in color)


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

compound_colors = {c: nice_colors[i] for i, c in enumerate(cfg.compounds)}


if os.path.exists("data_size.pkl"):
    data = pickle.load(open("data_size.pkl", "rb"))
else:
    data = []
    for compound in cfg.compounds:
        data.append(get_data_sizes(compound, "FEFF"))
    data.append(get_data_sizes("Cu", "VASP"))
    data.append(get_data_sizes("Ti", "VASP"))
data = np.array(data)


hatches = {
    "Anomalies": "xxxxx",
    "Unconverged": "/////",
    "ML": None,
}

import matplotlib as mpl

ASPECT_RATIO = 4 / 3
HEIGHT = 6
WEIGHT = HEIGHT * ASPECT_RATIO
DPI = 300
FONTSIZE = 14
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

# colors_ml = "#1b9e77"
# colors_anomalies = "#d95f02"
# colors_unconverged = "#7570b3"

tableau_colorblind10 = [
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
colors_ml = tableau_colorblind10[0]
colors_anomalies = tableau_colorblind10[1]
colors_unconverged = tableau_colorblind10[2]


cmap = "tab10"
compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))
compound_colors = {
    c: compound_colors[i]
    for i, c in enumerate(cfg.compounds + ["Ti\nVASP", "Cu\nVASP"])
}
hatches = {
    "ML": None,
    "Anomalies": "||",
    "Unconverged": "..",
}

# hatches = {
#     "universal_feff": "..",
#     "per_compound_tl": ".....",
#     "ft_tl": "",
# }

fig, ax = plt.subplots(1, 1, figsize=(WEIGHT, HEIGHT), dpi=DPI)
for i, compound in enumerate(cfg.compounds + ["Ti\nVASP", "Cu\nVASP"]):
    ax.bar(
        i,
        data[i]["ml"],
        # color=colors_ml,
        color=compound_colors[compound],
        hatch=hatches["ML"],
        label=compound,
        # edgecolor="black",
        # edgecolor=darken_color(compound_colors[compound.split("\n")[0]]),
        zorder=3,
        edgecolor="black",
    )
    ax.bar(
        i,
        data[i]["anomalies"],
        bottom=data[i]["ml"],
        # color=colors_anomalies,
        color=compound_colors[compound],
        # hatch=hatches["Anomalies"],
        alpha=0.6,
        # edgecolor="black",
        # fill=None,
        # hatch=hatches["Anomalies"],
        # edgecolor=darken_color(compound_colors[compound.split("\n")[0]]),
        zorder=3,
        edgecolor="black",
    )
    ax.bar(
        i,
        data[i]["unconverged"],
        bottom=data[i]["ml"] + data[i]["anomalies"],
        color=compound_colors[compound],
        # color=colors_unconverged,
        alpha=0.3,
        # edgecolor="black",
        # fill=None,
        # hatch=hatches["Unconverged"],
        # edgecolor=darken_color(compound_colors[compound.split("\n")[0]]),
        edgecolor="black",
        zorder=3,
        # hatch=hatches["Unconverged"],
    )
    ax.set_xticks(range(len(cfg.compounds) + 2))
    ax.set_xticklabels(cfg.compounds + ["Ti\nVASP", "Cu\nVASP"])

# add hatches for legend
from matplotlib.patches import Patch

# make VASP text of last two compounds bold

ax.set_xticklabels(
    cfg.compounds
    + [
        # use smaller font for vasp
        "Ti\n" + r"{\normalsize VASP}",
        "Cu\n" + r"{\normalsize VASP}",
    ],
)

ax.legend(
    [
        Patch(facecolor="white", edgecolor="black", hatch=hatches["Unconverged"]),
        Patch(facecolor="white", edgecolor="black", hatch=hatches["Anomalies"]),
        Patch(facecolor="white", edgecolor="black", hatch=hatches["ML"]),
    ],
    ["Unconverged", "Anomalies", "ML"],
    fontsize=FONTSIZE,
    frameon=True,
)

ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_xlabel("Compound", fontsize=FONTSIZE * 1.2, labelpad=-10)
ax.set_ylabel("Number of Spectra", fontsize=FONTSIZE * 1.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.tight_layout()
plt.savefig("data_size.pdf")


# %%
