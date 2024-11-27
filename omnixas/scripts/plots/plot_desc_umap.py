# %%
import datamapplot
import glasbey

import numpy as np
import pandas as pd
import umap.umap_ as umap

from matplotlib import colors as mcolors
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from config.defaults import cfg

# from omnixas.model.trained_xasblock import TrainedXASBlock

# %%

from omnixas.data import FEFFDataTags, MLSplits
from omnixas.utils.io import DEFAULTFILEHANDLER

for tag in FEFFDataTags():
    desc = pd.read_csv(f"dataset/descriptors/{tag.element}.csv")
    split = DEFAULTFILEHANDLER().deserialize_json(MLSplits, tag)
    print(tag.element, len(split), len(desc))

# %%


# %%
def get_desc_df(compound):
    df_desc = pd.read_csv(cfg.paths.descriptors.format(compound=compound))
    df_desc.columns = ["ids", "sites"] + list(df_desc.columns[2:])
    simulation_type = "FEFF"
    ml_data = np.load(
        cfg.paths.ml_data.format(compound=compound, simulation_type=simulation_type)
    )
    df_ml = pd.DataFrame(
        {
            "ids": ml_data["ids"],
            "sites": ml_data["sites"],
            "features": ml_data["features"].tolist(),
            "spectras": ml_data["spectras"].tolist(),
        }
    )
    df_ml.sites = df_ml.sites.astype("int64")
    df = pd.merge(df_desc, df_ml, on=["ids", "sites"])
    df["compound"] = compound
    return df


df = pd.concat([get_desc_df(c) for c in cfg.compounds])
df["compound_idx"] = pd.Categorical(
    df.compound,
    categories=cfg.compounds,
).codes


# %%

# %%

# FEATURES UMAP
desc_umap_features = {
    "n_components": 2,
    "n_neighbors": 50,
    "min_dist": 0.8,
    "random_state": 7,
}

umap_proj_features = umap.UMAP(**desc_umap_features).fit_transform(
    StandardScaler().fit_transform(df.features.tolist())
    # plot_df.features.tolist()
)
# %%

# SPECTRA UMAP
spectra_umap_kwargs = {
    "n_components": 2,
    "n_neighbors": 25,
    "min_dist": 0.8,
    "random_state": 7,
}
umap_proj_spectra = umap.UMAP(**spectra_umap_kwargs).fit_transform(
    StandardScaler().fit_transform(df.spectras.tolist())
    # plot_df.spectras.tolist()
)


# %%

plot_df = df.copy()
plot_df["umap_f0"] = umap_proj_features[:, 0]
plot_df["umap_f1"] = umap_proj_features[:, 1]
plot_df["umap_s0"] = umap_proj_spectra[:, 0]
plot_df["umap_s1"] = umap_proj_spectra[:, 1]


# %%

# PARAMETERS
DATA = "feature"  # ["feature", "spectra"]
DESC = "OCN"  # ["compound", "OS", "CN", "OCN", "NNRS", "MOOD]
ADD_LEGEND = False  # manually add legend
ADD_LABEL = False  # use lib function for labels
FONT_SIZE = 42 if DESC != "OCN" else 25
ADD_TITLE = True
ADD_YLABEL = False
DARK_MODE = True
ADD_GLOW = True  # True
POINT_SIZE = 0.75
ADD_BACKClR = False
BACKGROUND_COLOR = "#1F1F1F"
Glow_KEYWORD = {
    "kernel": "gaussian",  # gaussian, tophat, epanechnikov, exponential, linear, cosine
    "kernel_bandwidth": 0.5 if DESC == "OCN" else 0.75,  # default: 0.25,
    "approx_patch_size": 16,  # default: 64
}

# DEFAULT PARAMETERS
datamapplot_kwargs = {
    # "palette_min_lightness": 0,
    # "palette_hue_shift": 0,
    # "palette_hue_radius_dependence": 0.1,
    "color_label_text": True,
    "color_label_arrows": True,
    "label_over_points": True,
    "dynamic_label_size_scaling_factor": 0.5,
    "dynamic_label_size": False,
    "max_font_size": 50,
    "min_font_size": 49,
    "font_family": "Urbanist",
    "verbose": True,
    "darkmode": False,  # label color (and prolly others) also gets affects
    # "alpha": 1,
    "add_glow": ADD_GLOW,  # SET TO FALSE FOR TESTING, glow makes diffused colors
    # "marker_color_array": [compound_colors[compound] for compound in labels],
    "dpi": 300,
    "noise_label": "nan",
    "use_medoids": True,  # better but slower
    "point_size": POINT_SIZE,
    "arrowprops": {
        "arrowstyle": "wedge,tail_width=0.1,shrink_factor=0.5",
        "connectionstyle": "arc3,rad=0.05",
        "linewidth": 0,
    },
    "label_font_size": FONT_SIZE,
    "label_base_radius": 10,
    # "title" = DESC,
}
# CUSTOMIZATIONS
datamapplot_kwargs["darkmode"] = DARK_MODE
datamapplot_kwargs["glow_keywords"] = Glow_KEYWORD
if ADD_LABEL is False:
    datamapplot_kwargs["label_font_size"] = 0  # remove labels


# # SLOW/BETTER
# datamapplot_kwargs["add_glow"] = True
# datamapplot_kwargs["use_medoids"] = True

datamapplot_kwargs["alpha"] = 1

# if DESC == "OCN":
#     datamapplot_kwargs["label_over_points"] = False
#     if DATA == "feature":
#         datamapplot_kwargs["label_direction_bias"] = 2
#         datamapplot_kwargs["label_base_radius"] = 20
#         datamapplot_kwargs["label_over_points"] = False
#     elif DATA == "spectra":
#         datamapplot_kwargs["label_direction_bias"] = 4  # makes label more vertical
#         datamapplot_kwargs["label_base_radius"] = 19


# less thatn 5% of maximum count
COUNT_THRESHOLD = 0.05 * plot_df[DESC].value_counts().max()
plot_df[DESC] = plot_df[DESC].replace(
    plot_df[DESC].value_counts()[plot_df[DESC].value_counts() < COUNT_THRESHOLD].index,
    np.nan,
)
labels = plot_df[DESC]

if DESC in ["OS", "CN"]:
    labels = labels.fillna(9999).astype("int64").astype("str").replace("9999", "nan")

if DESC == "OCN":
    labels = (
        labels.fillna(9999)
        .apply(lambda x: np.round(x * 2) / 2)  # Round to nearest 0.5
        .apply(
            lambda x: int(x) if x.is_integer() else x
        )  # Convert to int if whole number
        .astype(str)
        .replace("9999.0", "nan")
    )


if DESC in ["OS", "CN", "OCN"]:
    if DESC == "OCN":
        # sort by labels and then use jet
        pallete = plt.get_cmap("Spectral")(np.linspace(0, 1, len(labels.unique())))
        pallete = glasbey.extend_palette(pallete, len(labels.unique()))
        color_dict = {}
        for i, label in enumerate(sorted(labels.unique())):
            if label == "nan":
                color_dict[label] = "#999999"  # default grey in datamapplot
            else:
                color_dict[label] = pallete[i - 1]

    else:
        pallete = glasbey.create_palette(
            palette_size=len(labels.unique()),
            lightness_bounds=(50, 80),  # smaller is darker
            chroma_bounds=(60, 100),  # smaller is less colorful
            # colorblind_safe=True,
        )
        color_dict = {}
        for i, label in enumerate(labels.unique()):
            if label == "nan":
                color_dict[label] = "#999999"  # default grey in datamapplot
            else:
                color_dict[label] = pallete[i - 1]

    # for i, label in enumerate(labels.unique()):

if DESC == "compound":
    color_dict = {
        c: mcolors.to_hex(plt.get_cmap("tab10")(i)) for i, c in enumerate(cfg.compounds)
    }
    # datamapplot_kwargs["title"] = "Element"

X = f"umap_{DATA[0].lower()}0"
Y = f"umap_{DATA[0].lower()}1"
points = np.array([plot_df[X].values, plot_df[Y].values]).T

fig, ax = datamapplot.create_plot(
    points,
    labels,
    marker_color_array=(
        [color_dict[compound] for compound in labels] if color_dict else None
    ),
    label_color_map=color_dict,
    highlight_labels=set(labels),
    **datamapplot_kwargs,
)

if ADD_BACKClR:
    ax.set_facecolor(BACKGROUND_COLOR)
    fig.patch.set_facecolor("#696969")


y_label = DESC if DESC != "compound" else "Element"
ax.set_ylabel(y_label, fontsize=32, fontweight="bold", color="white")

ax.set_title(
    "Features" if DATA == "feature" else "Spectras",
    fontsize=32,
    fontweight="bold",
    color="white",
)

if ADD_TITLE:
    ax.text(
        0.5,
        0.95,
        f"{DESC}" if DESC != "compound" else "Element",
        fontsize=44,
        color="#AAAAAA",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontweight="bold",
        fontname="DejaVu Sans",
    )

if ADD_YLABEL:
    ax.text(
        0.05,
        0.5,
        DATA.capitalize(),
        fontsize=44,
        color="#AAAAAA",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontweight="bold",
        fontname="DejaVu Sans",
        rotation=90,
    )


if ADD_LEGEND:
    i = 0
    color_dict = dict(sorted(color_dict.items(), key=lambda item: item[0]))
    count = len(color_dict) - 1  # remove nan
    dx = 0.75 / count if DESC != "OCN" else 0.9 / count
    dy = 0.2 / FONT_SIZE
    max_row = 9
    x0 = 0.5 - dx * count / 2 + dx / 2
    y0 = 0.075
    for label, color in color_dict.items():
        if label == "nan":
            continue
        i += 1
        row = (i - 1) // max_row
        col = (i - 1) % max_row
        x = x0 + col * dx
        y = y0 - row * dy
        ax.text(
            x,
            y,
            f"{label}",
            fontsize=FONT_SIZE,
            color=color,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontweight="bold",
            # font should be math font
            fontname="DejaVu Sans",
        )


fig.tight_layout()
# fig.savefig(f"umap_{DESC}_{DATA}.png", bbox_inches="tight", dpi=300)
fig.savefig(f"umap_{DESC}_{DATA}.pdf", bbox_inches="tight", dpi=300)
