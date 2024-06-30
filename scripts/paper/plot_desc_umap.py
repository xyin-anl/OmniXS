# %%

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


def get_balanced_df_with_unique_ids():
    df = pd.concat([get_desc_df(c) for c in cfg.compounds])
    df_unique_dict = {
        c: df[df.compound == c].drop_duplicates(subset="ids", keep="first")
        for c in cfg.compounds
    }
    COUNT = min([len(d) for d in df_unique_dict.values()])
    df_unique = pd.concat([d.sample(COUNT) for d in df_unique_dict.values()])
    df_unique["compound_idx"] = pd.Categorical(
        df_unique.compound, categories=cfg.compounds
    ).codes
    return df_unique


df_unique = get_balanced_df_with_unique_ids()

# %%

# UMAP OF FEATURES
umap_proj = umap.UMAP(
    n_components=2, n_neighbors=25, min_dist=0.45, random_state=7
).fit_transform(df_unique.features.tolist())
df_unique["umap_f0"] = umap_proj[:, 0]
df_unique["umap_f1"] = umap_proj[:, 1]

# %%
# UMAP OF SPECTRAS
umap_proj = umap.UMAP(
    n_components=2, n_neighbors=25, min_dist=0.45, random_state=7
).fit_transform(df_unique.spectras.tolist())
df_unique["umap_s0"] = umap_proj[:, 0]
df_unique["umap_s1"] = umap_proj[:, 1]


# %%


DESCRIPTORS = ["OS", "CN", "OCN", "NNRS", "MOOD"]
color_by = ["compound_idx", "OS", "CN", "OCN", "NNRS", "MOOD"]
cmap_dict = {
    "compound_idx": "tab10",
    "OS": "jet",
    "CN": "jet",
    "OCN": "jet",
    "NNRS": "jet",
    "MOOD": "jet",
}
FONTSIZE = 22
X, Y = "umap_f0", "umap_f1"
# X, Y = "umap_s0", "umap_s1"


def series_to_color(s, cmap="jet"):
    """Utility function to convert a series to a color array for plotting."""
    select_idx = ~s.isna() & ~np.isinf(s)
    if cmap != "tab10":
        normalizer = Normalize()
        non_outliers = plt.cm.get_cmap(cmap)(normalizer(s[select_idx]))
    else:
        non_outliers = plt.cm.get_cmap(cmap)(s[select_idx])
    c = np.zeros((len(s), 4))
    c[select_idx] = non_outliers
    c[~select_idx] = (0, 0, 0, 0)
    return c


fig = plt.figure(figsize=(15, 20))
plt.style.use(["default", "science"])
gs = fig.add_gridspec((len(color_by) + 1) // 2, 2, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for ax, d in zip(axs.flatten(), color_by):

    # # COLOR EACH COMPOUND BLOCK AT A TIME
    # for c in cfg.compounds:
    #     ax.scatter(
    #         df_unique[df_unique.compound == c][X],
    #         df_unique[df_unique.compound == c][Y],
    #         c=series_to_color(df_unique[df_unique.compound == c][d], cmap=cmap_dict[d]),
    #         s=0.8,
    #         # alpha=0.8,
    #         label=c,
    #     )
    # fig.suptitle("Local color scaling", fontsize=FONTSIZE * 1.2, y=0.9)
    # filename = f"umap_desc_{X}_{Y}_local.pdf"

    # COLOR ALL POINTS AT ONCE
    ax.scatter(
        df_unique[X],
        df_unique[Y],
        c=series_to_color(df_unique[d], cmap=cmap_dict[d]),
        s=2,
        alpha=0.8,
    )
    fig.suptitle("Global color scaling", fontsize=FONTSIZE * 1.2, y=0.9)
    filename = f"umap_desc_{X}_{Y}_global.pdf"

    # title top top left inside of the axes
    if d != "compound_idx":
        ax.set_title(d, fontsize=FONTSIZE, loc="left", x=0.01, y=0.9)
    else:
        ax.set_title("Element", fontsize=FONTSIZE, loc="left", x=0.01, y=0.9)
    ax.set_xticks([])
    ax.set_yticks([])


fig.savefig(filename, dpi=500, bbox_inches="tight")

# %%
df_unique.to_csv("umap_descriptors.csv", index=False)
