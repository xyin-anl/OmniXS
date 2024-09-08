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
plot_df = df.copy()

# %%


def series_to_color(s, cmap="jet"):
    valid_points = deepcopy(s)
    valid_points = ~valid_points.isna()
    valid_points &= ~np.isinf(valid_points)

    # cmap is dict
    if isinstance(cmap, dict):
        # value_counts = plot_df[d].value_counts()
        value_counts = s.value_counts()
        frequent_values = value_counts[value_counts > 100].index
        # valid_points &= plot_df[d].isin(frequent_values)
        valid_points &= s.isin(frequent_values)
        # c = np.repeat("#00000000", len(s))
        c = np.repeat("#00000000", len(s))
        for value, color in cmap.items():
            if value in frequent_values:
                # c[plot_df[d] == value] = color
                c[s == value] = color
    elif cmap == "tab10":
        c = np.zeros((len(s), 4))
        c[valid_points] = plt.get_cmap(cmap)(s[valid_points])
    else:
        c = np.zeros((len(s), 4))
        norm = Normalize(vmin=s[valid_points].min(), vmax=s[valid_points].max())
        c = np.zeros((len(s), 4))
        c[valid_points] = plt.get_cmap(cmap)(norm(s[valid_points]))
    return c


X, Y = "umap_s0", "umap_s1"
# X, Y = "umap_f0", "umap_f1"
DESCRIPTORS = ["OS", "CN", "OCN", "NNRS", "MOOD"]

discrete_colors = {
    "1": "#9932CC",  # DarkOrchid
    "3": "#FFD700",  # Gold
    "2": "#DC143C",  # Crimson
    "4": "#00BFFF",  # DeepSkyBlue
    "5": "#32CD32",  # LimeGreen
    "6": "#FF4500",  # OrangeRed
    "8": "#800000",  # Maroon
    "nan": "#00000000",
}


# X, Y = "umap_s0", "umap_s1"
FONTSIZE = 22
cmap_dict = {
    "compound_idx": "tab10",
    "OS": discrete_colors,
    "CN": discrete_colors,
    "OCN": "jet",
    "NNRS": "jet",
    "MOOD": "jet",
}


plot_DESCIPTORS = DESCRIPTORS[:-1]
cols = ["feature", "spectra"] * len(plot_DESCIPTORS)
rows = np.repeat(["compound_idx"] + plot_DESCIPTORS, 2)
pairs = list(zip(cols, rows))


fig = plt.figure(figsize=(16, 30))
gs = fig.add_gridspec(len(pairs) // 2, 2, hspace=0, wspace=0)
axs = gs.subplots()
for ax, (data_class, descriptor) in zip(axs.flatten(), pairs):
    X = f"umap_{data_class[0].lower()}0"
    Y = f"umap_{data_class[0].lower()}1"
    d = descriptor

    SAMPLES = 5000
    ax.scatter(
        plot_df[X].sample(SAMPLES, random_state=7),
        plot_df[Y].sample(SAMPLES, random_state=7),
        c=series_to_color(
            pd.Series(plot_df[d].sample(SAMPLES, random_state=7), name=d),
            cmap=cmap_dict[d],
        ),
        # s=0.05,
    )

    # ax.scatter(
    #     plot_df[X],
    #     plot_df[Y],
    #     c=series_to_color(plot_df[d], cmap=cmap_dict[d]),
    #     s=0.05,
    # )

    # add grid
    if data_class == "spectra":
        ax.set_xticks(np.linspace(-5, 30, 8))
    ax.grid(True)

    # if cmap_dict[d] == discrete_colors:
    #     if cmap_dict[d] == discrete_colors:
    #         create_legend(ax, d, discrete_colors)

    # if ax == axs[-1, -1]:
    #     ax.scatter(
    #         plot_df[X],
    #         plot_df[Y],
    #         c=plot_df[d],
    #         cmap="jet",
    #         s=0.05,
    #         norm=Normalize(vmin=plot_df[d].min(), vmax=plot_df[d].max()),
    #     )

    # if ax == axs[1, 0]:
    #     break
    #     # continue

# Assume 'fig' is your figure object and 'ax' is your main axes

# Get the position of the main axes
pos = ax.get_position()
cax = fig.add_axes([pos.x0 + 0.045, pos.y0 - 0.02, pos.width - 0.09, 0.01])

cbar = plt.colorbar(
    # ax.collections[0],
    axs[-1, -1].collections[0],
    cax=cax,
    orientation="horizontal",
    # fraction=0.05,
)
cbar.set_label(
    r"OCN",
    fontsize=FONTSIZE,
    fontweight="bold",
    rotation=0,
    labelpad=10,  # Adjust this value as needed
)

# fig.savefig("umap_desc.pdf", bbox_inches="tight", dpi=300)

# %%

# plot all descriptors in one plot
# color_by = ["compound_idx", "OS", "CN", "OCN", "NNRS", "MOOD"]
# X, Y = "umap_s0", "umap_s1"
# fig = plt.figure(figsize=(15, 20))
# plt.style.use(["default", "science"])
# gs = fig.add_gridspec((len(color_by) + 1) // 2, 2, hspace=0, wspace=0)
# axs = gs.subplots(sharex=True, sharey=True)
# for ax, d in zip(axs.flatten(), color_by):
#     ax.scatter(
#         plot_df[X],
#         plot_df[Y],
#         c=series_to_color(plot_df[d], cmap=cmap_dict[d]),
#         s=0.05,
#         # alpha=0.5,
#     )
#     fig.suptitle("Global color scaling", fontsize=FONTSIZE * 1.2, y=0.9)
#     filename = f"umap_desc_{X}_{Y}_global.pdf"
#     # title top top left inside of the axes
#     if d != "compound_idx":
#         ax.set_title(d, fontsize=FONTSIZE, loc="left", x=0.01, y=0.9)
#     else:
#         ax.set_title("Element", fontsize=FONTSIZE, loc="left", x=0.01, y=0.9)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # break

# LEGECY
# def get_balanced_df_with_unique_ids():
#     df = pd.concat([get_desc_df(c) for c in cfg.compounds])
#     df_unique_dict = {
#         c: df[df.compound == c].drop_duplicates(subset="ids", keep="first")
#         for c in cfg.compounds
#     }
#     COUNT = min([len(d) for d in df_unique_dict.values()])
#     df_unique = pd.concat([d.sample(COUNT) for d in df_unique_dict.values()])
#     df_unique["compound_idx"] = pd.Categorical(
#         df_unique.compound, categories=cfg.compounds
#     ).codes
#     return df_unique
# df_unique = get_balanced_df_with_unique_ids()

# # PARMAS USED IN FIRST PLOT IN PAPER
# umap_proj = umap.UMAP(
#     n_components=2,
#     n_neighbors=25,
#     min_dist=0.45,
#     random_state=7,
# ).fit_transform(df_unique.features.tolist())
# umap_proj = umap.UMAP(
#     n_components=2,
#     n_neighbors=25,
#     min_dist=0.45,
#     random_state=7,
# ).fit_transform(df_unique.features.tolist())
