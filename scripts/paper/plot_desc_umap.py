# %%
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
## %%


# %%

#
plot_df = df.copy()
plot_df["umap_f0"] = umap_proj_features[:, 0]
plot_df["umap_f1"] = umap_proj_features[:, 1]
plot_df["umap_s0"] = umap_proj_spectra[:, 0]
plot_df["umap_s1"] = umap_proj_spectra[:, 1]


# %%

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
    "add_glow": False,  # SET TO FALSE FOR TESTING, glow makes diffused colors
    # "marker_color_array": [compound_colors[compound] for compound in labels],
    "dpi": 300,
    "noise_label": "nan",
    "use_medoids": True,  # better but slower
    "point_size": 3,
    "arrowprops": {
        "arrowstyle": "wedge,tail_width=0.1,shrink_factor=0.5",
        "connectionstyle": "arc3,rad=0.05",
        "linewidth": 0,
    },
    "label_font_size": 50,
    "label_base_radius": 10,
}


# PARAMETERS
data_class = "spectra"
desc = "compound"  # ["compound_idx", "OS", "CN", "OCN", "NNRS", "MOOD]
datamapplot_kwargs["darkmode"] = True

# # # FAST/OK
# # datamapplot_kwargs["add_glow"] = False
# # datamapplot_kwargs["use_medoids"] = False

# SLOW/BETTER
datamapplot_kwargs["add_glow"] = True
datamapplot_kwargs["use_medoids"] = True

datamapplot_kwargs["alpha"] = 1
if desc == "OCN":
    datamapplot_kwargs["label_over_points"] = False
    if data_class == "feature":
        datamapplot_kwargs["label_direction_bias"] = 2
        datamapplot_kwargs["label_base_radius"] = 20
        datamapplot_kwargs["label_over_points"] = False
    elif data_class == "spectra":
        datamapplot_kwargs["label_direction_bias"] = 4  # makes label more vertical
        datamapplot_kwargs["label_base_radius"] = 19


# less thatn 5% of maximum count
COUNT_THRESHOLD = 0.05 * plot_df[desc].value_counts().max()
plot_df[desc] = plot_df[desc].replace(
    plot_df[desc].value_counts()[plot_df[desc].value_counts() < COUNT_THRESHOLD].index,
    np.nan,
)
labels = plot_df[desc]

if desc in ["OS", "CN"]:
    labels = labels.fillna(9999).astype("int64").astype("str").replace("9999", "nan")

if desc == "OCN":
    labels = (
        labels.fillna(9999)
        .apply(lambda x: np.round(x * 2) / 2)  # Round to nearest 0.5
        .apply(
            lambda x: int(x) if x.is_integer() else x
        )  # Convert to int if whole number
        .astype(str)
        .replace("9999.0", "nan")
    )


if desc in ["OS", "CN", "OCN"]:
    glasbey_palette = glasbey.create_palette(
        palette_size=len(labels.unique()),
        # lightness_bounds=(50, 100),
        # chroma_bounds=(40, 80),
        # colorblind_safe=True,
    )
    # datamapplot_kwargs["title"] = desc
    color_dict = {}
    for i, label in enumerate(labels.unique()):
        if label == "nan":
            color_dict[label] = "#999999"  # default grey in datamapplot
        else:
            color_dict[label] = glasbey_palette[i - 1]

if desc == "compound":
    color_dict = {
        c: mcolors.to_hex(plt.get_cmap("tab10")(i)) for i, c in enumerate(cfg.compounds)
    }
    # datamapplot_kwargs["title"] = "Element"


X = f"umap_{data_class[0].lower()}0"
Y = f"umap_{data_class[0].lower()}1"
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

ax.set_facecolor("#1F1F1F")
fig.patch.set_facecolor("#696969")


y_label = desc if desc != "compound" else "Element"
ax.set_ylabel(y_label, fontsize=32, fontweight="bold", color="white")

ax.set_title(
    "Features" if data_class == "feature" else "Spectras",
    fontsize=32,
    fontweight="bold",
    color="white",
)

fig.tight_layout()
fig.savefig(f"umap_{desc}_{data_class}.png", bbox_inches="tight", dpi=300)
fig.savefig(f"umap_{desc}_{data_class}.pdf", bbox_inches="tight", dpi=300)


# %%
