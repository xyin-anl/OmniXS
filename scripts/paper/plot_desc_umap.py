# %%
import torch
import datamapplot
import glasbey
from matplotlib import colors as mcolors
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import scienceplots
import umap.umap_ as umap
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_all_data, load_xas_ml_data
from src.models.trained_models import Trained_FCModel
import pandas as pd

# %%


# LOAD ALL DATA
def get_desc_df(compound):
    df_desc = pd.read_csv(cfg.paths.descriptors.format(compound=compound))
    df_desc.columns = ["ids", "sites"] + list(df_desc.columns[2:])
    df_desc.columns
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
ADD_GLOW = True # True
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

        # datamapplot_kwargs["add_glow"] = False  # Gives error otherwise for some reason

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


# %%

# compound = "Cu"
# simulation_type = "FEFF"
# ml_data = np.load(
#     cfg.paths.ml_data.format(compound=compound, simulation_type=simulation_type)
# )
# ml_data.keys()
# df_ml = pd.DataFrame(
#     {
#         "ids": ml_data["ids"],
#         "sites": ml_data["sites"],
#         "features": ml_data["features"].tolist(),
#         "spectras": ml_data["spectras"].tolist(),
#     }
# )
# # merge plot_df with df_ml based on ids and sites
# df_ml.sites = df_ml.sites.astype("int64")
# df_mse = pd.merge(df, df_ml, on=["ids", "sites"])

# %%

plot_df = plot_df.copy()

simulation_type = "FEFF"
model_name = "per_compound_tl"
compound = "Cu"

# models = {
#     compound: Trained_FCModel(
#         DataQuery(compound, simulation_type), name=model_name
#     ).model.to("mps")
#     for compound in cfg.compounds
# }
# # model = Trained_FCModel(DataQuery(compound, simulation_type), name=model_name).model

models = {
    model_name: {
        compound: Trained_FCModel(
            DataQuery(compound, simulation_type), name=model_name
        ).model.to("mps")
        for compound in cfg.compounds
    }
    for model_name in ["per_compound_tl", "ft_tl"]
}


# %%

for compound in cfg.compounds:
    features = plot_df[plot_df.compound == compound]["features"]
    features = features.apply(lambda x: np.array(x) * 1000).tolist()
    features = torch.tensor(features).float().to("mps")
    # features = torch.tensor(features.tolist()).float().to("mps")
    predictions = models[compound](features).cpu().detach().numpy()
    truth = plot_df[plot_df.compound == compound]["spectras"].tolist()
    truth = np.array(truth) * 1000
    mse = np.mean((predictions - truth) ** 2, axis=1)
    plot_df.loc[plot_df.compound == compound, "mse"] = mse

    # plt.hist(np.log(mse), bins=100, alpha=0.5)
    # plt.title(compound)
    # plt.show()

# %%

# mse_diff based on "per_compound_tl" model and "ft_tl" model
for compound in cfg.compounds:
    for model_name in ["per_compound_tl", "ft_tl"]:
        features = plot_df[plot_df.compound == compound]["features"]
        features = features.apply(lambda x: np.array(x) * 1000).tolist()
        features = torch.tensor(features).float().to("mps")
        predictions = models[model_name][compound](features).cpu().detach().numpy()
        truth = plot_df[plot_df.compound == compound]["spectras"].tolist()
        truth = np.array(truth) * 1000
        mse = np.mean((predictions - truth) ** 2, axis=1)
        plot_df.loc[plot_df.compound == compound, f"mse_{model_name}"] = mse

# add one for universal_tl
features = plot_df["features"]
features = features.apply(lambda x: np.array(x) * 1000).tolist()
features = torch.tensor(features).float().to("mps")
model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl").model.to("mps")
predictions = model(features).cpu().detach().numpy()
truth = plot_df["spectras"].tolist()
truth = np.array(truth) * 1000
mse = np.mean((predictions - truth) ** 2, axis=1)
plot_df["mse_universal_tl"] = mse

# %%
# add mse diff between per_compound_tl and ft_tl
plot_df["mse_diff"] = +plot_df["mse_ft_tl"] - plot_df["mse_per_compound_tl"]

# %%


color_groups = [
    "OS",
    "CN",
    "OCN",
    "mse_per_compound_tl",
    "mse_universal_tl",
    "mse_ft_tl",
    "mse_diff",
]
# color_groups = ["OS", "CN", "OCN", "NNRS", "mse"]
fig = plt.figure(figsize=(4 * len(color_groups), 4 * len(cfg.compounds)))
gs = fig.add_gridspec(len(cfg.compounds), len(color_groups), hspace=0, wspace=0)
for i, compound in enumerate(cfg.compounds):
    plot_group = plot_df[plot_df.compound == compound]
    for j, color_group in enumerate(color_groups):
        ax = fig.add_subplot(gs[i, j])

        if "mse" in color_group and color_group != "mse_diff":
            color_normalizer = PowerNorm(
                gamma=0.2,
                vmin=plot_group[color_group].min(),
                vmax=plot_group[color_group].max(),
            )
            colors = color_normalizer(plot_group[color_group])
        elif color_group in ["mse_diff"]:
            # it has negative values
            color_normalizer = SymLogNorm(
                linthresh=0.00001,
                linscale=0.1,
                vmin=plot_group[color_group].min(),
                vmax=plot_group[color_group].max(),
            )
            colors = color_normalizer(plot_group[color_group])
        else:

            # colors = plt.get_cmap("tab10")(plot_group[color_group].astype("int"))
            # handle nan
            colors = plot_group[color_group].replace("nan", np.nan)
            colors = colors.fillna(9999).astype("int")
            colors = plt.get_cmap("tab10")(colors)

        ax.scatter(
            plot_group.umap_s0,
            plot_group.umap_s1,
            # plot_group.umap_f0,
            # plot_group.umap_f1,
            c=colors,
            s=1,
            cmap="jet",
        )
        ax.set_title(f"{compound} {color_group}")
        ax.axis("off")
        ax.set_aspect("equal")

fig.tight_layout()
plt.savefig("mse_umap.png", dpi=300)

# %%

fig = plt.figure(figsize=(4 * len(color_groups), 4))
gs = fig.add_gridspec(1, len(color_groups), hspace=0, wspace=0)
for j, color_group in enumerate(color_groups):
    ax = fig.add_subplot(gs[0, j])
    if "mse" in color_group and color_group != "mse_diff":
        color_normalizer = PowerNorm(
            gamma=0.2,
            vmin=plot_df[color_group].min(),
            vmax=plot_df[color_group].max(),
        )
        colors = color_normalizer(plot_df[color_group])
    elif color_group in ["mse_diff"]:
        # it has negative values
        color_normalizer = SymLogNorm(
            linthresh=0.00001,
            linscale=0.1,
            vmin=plot_df[color_group].min(),
            vmax=plot_df[color_group].max(),
        )
        colors = color_normalizer(plot_df[color_group])
    else:
        # colors = plt.get_cmap("tab10")(plot_group[color_group].astype("int"))
        # handle nan
        colors = plot_df[color_group].replace("nan", np.nan)
        colors = colors.fillna(9999).astype("int")
        colors = plt.get_cmap("tab10")(colors)

    ax.scatter(
        plot_df.umap_s0,
        plot_df.umap_s1,
        # plot_df.umap_f0,
        # plot_df.umap_f1,
        c=colors,
        s=1,
        cmap="jet",
    )
    ax.set_title(f"{color_group}")
    ax.axis("off")
    ax.set_aspect("equal")
fig.tight_layout()
plt.savefig("mse_umap_all.png", dpi=300)

# %%
