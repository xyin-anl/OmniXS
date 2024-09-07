# %%
import matplotlib as mpl
import warnings
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
import matplotlib.ticker as ticker

# %%


def get_data_sizes(compound, simulation_type):

    def data_count(ml_splits):
        return len(ml_splits.train.X) + len(ml_splits.val.X) + len(ml_splits.test.X)

    pre_filter_size = data_count(
        load_xas_ml_data(
            DataQuery(compound, simulation_type),
            filter_spectra_anomalies=False,
            use_cache=False,
        )
    )
    post_filter_size = data_count(
        load_xas_ml_data(
            DataQuery(compound, simulation_type),
            filter_spectra_anomalies=True,
            use_cache=False,
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


if os.path.exists("data_size.pkl"):
    warnings.warn("Using cached data")
    data = pickle.load(open("data_size.pkl", "rb"))
else:
    data = []
    for compound in cfg.compounds:
        data.append(get_data_sizes(compound, "FEFF"))
    data.append(get_data_sizes("Cu", "VASP"))
    data.append(get_data_sizes("Ti", "VASP"))
    pickle.dump(data, open("data_size.pkl", "wb"))
data = np.array(data)


# %%


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def darken_color(color, factor=0.4):
    if isinstance(color, str):
        color = hex_to_rgb(color)
    return tuple(max(0, min(1, c * factor)) for c in color)


FONTSIZE = 20
hatches = {
    "Anomalies": "xxxxx",
    "Unconverged": "/////",
    "ML": None,
}


colors_ml = plt.get_cmap("tab10")(0)
colors_anomalies = plt.get_cmap("tab10")(1)
colors_unconverged = plt.get_cmap("tab10")(2)
compound_colors = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(cfg.compounds)}


plt.style.use(["default", "science"])
compound_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(cfg.compounds) + 2))
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

ASPECT_RATIO = 4 / 3
HEIGHT = 6
WEIGHT = HEIGHT * ASPECT_RATIO
fig, ax = plt.subplots(1, 1, figsize=(WEIGHT, HEIGHT), dpi=300)
for i, compound in enumerate(cfg.compounds + ["Ti\nVASP", "Cu\nVASP"]):
    ax.bar(
        i,
        data[i]["ml"],
        # color=colors_ml,
        color=compound_colors[compound],
        hatch=hatches["ML"],
        label=compound,
        # edgecolor="black",
        edgecolor=darken_color(compound_colors[compound.split("\n")[0]]),
        zorder=3,
        # edgecolor="black",
    )
    ax.bar(
        i,
        data[i]["anomalies"],
        bottom=data[i]["ml"],
        # color=colors_anomalies,
        color=compound_colors[compound],
        hatch=hatches["Anomalies"],
        alpha=0.8,
        # edgecolor="black",
        # fill=None,
        # hatch=hatches["Anomalies"],
        edgecolor=darken_color(compound_colors[compound.split("\n")[0]]),
        # edgecolor="black",
        zorder=3,
        # edgecolor="black",
    )
    ax.bar(
        i,
        data[i]["unconverged"],
        bottom=data[i]["ml"] + data[i]["anomalies"],
        color=compound_colors[compound],
        # color=colors_unconverged,
        alpha=0.55,
        # edgecolor="black",
        # fill=None,
        hatch=hatches["Unconverged"],
        edgecolor=darken_color(compound_colors[compound.split("\n")[0]]),
        # edgecolor="black",
        zorder=3,
        # hatch=hatches["Unconverged"],
    )
    ax.set_xticks(range(len(cfg.compounds) + 2))
    ax.set_xticklabels(
        cfg.compounds + ["Ti\nVASP", "Cu\nVASP"], fontsize=FONTSIZE * 0.8
    )

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(3, 3))
    ax.tick_params(axis="y", which="major", labelsize=FONTSIZE * 0.8)
    ax.yaxis.get_offset_text().set_size(FONTSIZE * 0.8)


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
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.tight_layout()
plt.savefig("data_size.pdf")


# %%
