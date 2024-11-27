# %%
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.patches import Patch

from omnixas.data import (
    DataTag,
    MLData,
    MLSplits,
)
from omnixas.utils.constants import ElementsFEFF, ElementsVASP, SpectrumType
from omnixas.utils.constants import Element
from omnixas.utils import DEFAULTFILEHANDLER, FileHandler
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data_raw import RAWDataVASP
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from omnixas.utils.constants import ElementsFEFF


# %%

# make cfg dictconfig with compounds given by cfg.compounds from ElementsFEFF
from omegaconf import OmegaConf, DictConfig

compounds = [str(e) for e in ElementsFEFF]
cfg = DictConfig(
    {
        "compounds": compounds,
    }
)

# %%


def get_data_sizes(element: Element, type: SpectrumType):
    data_tag = DataTag(element=element, type=type)

    raw_data = (
        RAWDataFEFF(element, type)
        if type == SpectrumType.FEFF
        else RAWDataVASP(element)
    )
    if type == SpectrumType.VASP:
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

    new_config = deepcopy(DEFAULTFILEHANDLER().config)
    new_config["MLData"][
        "directory"
    ] = "dataset/before_outlier_removal/{type}/{element}"
    pre_filter_size = len(
        list(
            FileHandler(new_config).fetch_serialized_objects(MLData, **data_tag.dict())
        )
    )
    post_filter_size = len(
        DEFAULTFILEHANDLER().deserialize_json(MLSplits, supplemental_info=data_tag)
    )
    anomalies = pre_filter_size - post_filter_size

    print(f"Compound: {element}, Simulation Type: {type}")
    print(f"RAW Data Size: {raw_data.total_sites}")
    print(f"Unconverged: {unconverged_count}")
    print(f"Missing: {len(raw_data.missing_data)}")
    print(f"Pre-Filter Size: {pre_filter_size}")
    print(f"Anomalies: {anomalies}")
    print(f"Post-Filter Size: {post_filter_size}")

    return {
        "total": len(raw_data),
        "unconverged": unconverged_count,
        "anomalies": anomalies,
        "ml": post_filter_size,
    }


DATA_COUNT = [get_data_sizes(element, SpectrumType.FEFF) for element in ElementsFEFF]
DATA_COUNT += [get_data_sizes(element, SpectrumType.VASP) for element in ElementsVASP]


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
        DATA_COUNT[i]["ml"],
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
        DATA_COUNT[i]["anomalies"],
        bottom=DATA_COUNT[i]["ml"],
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
        DATA_COUNT[i]["unconverged"],
        bottom=DATA_COUNT[i]["ml"] + DATA_COUNT[i]["anomalies"],
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
