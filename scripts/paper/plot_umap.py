# %%

from matplotlib.colors import Normalize, LogNorm, SymLogNorm, PowerNorm

# do tsne
from sklearn.manifold import TSNE

import numpy as np
import scienceplots
import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_all_data, load_xas_ml_data
from src.models.trained_models import Trained_FCModel

# %%

data, compound_names = load_all_data(return_compound_name=True, sample_method="all")
colors = plt.cm.tab10.colors[:8]
colors = {c: col for c, col in zip(cfg.compounds, colors)}
plt_colors = [colors[k] for k in compound_names[2]]
train_colors = [colors[k] for k in compound_names[0]]
val_colors = [colors[k] for k in compound_names[1]]
test_colors = [colors[k] for k in compound_names[2]]

# %%


def plot_umap_1(
    data,
    n_neighbors=20,
    min_dist=0.3,
    umap_data=None,
    seed=7,
):
    if umap_data is None:
        umap_data = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        ).fit_transform(data)


# %%

umap_data = np.concatenate([data.train.X, data.val.X, data.test.X])
umap_plt_colors = np.concatenate([train_colors, val_colors, test_colors])

n_neighbors = 50
min_dist = 0.3
seed = 7
umap_proj = umap.UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    random_state=seed,
).fit_transform(umap_data)

# %%

FONTSIZE = 18
fig, ax = plt.subplots(figsize=(8, 8))
plt.style.use(["default", "science"])
ax.scatter(
    umap_proj[:, 0],
    umap_proj[:, 1],
    c=umap_plt_colors,
    s=0.05,
    # alpha=0.1,
)
# ax.set_title(f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed}")
# ax.set_xlabel("UMAP Dim 1", fontsize=FONTSIZE)
# ax.set_ylabel("UMAP Dim 2", fontsize=FONTSIZE)

# ax bacground color to be very light gray
ax.set_facecolor("#f0f0f0")

# ax.legend(
#     [
#         plt.Line2D(
#             [0],
#             [0],
#             marker="o",
#             color="w",
#             markerfacecolor=col,
#             markersize=10,
#         )
#         for col in plt.cm.tab10.colors[:8]
#     ],
#     cfg.compounds,
#     fontsize=FONTSIZE * 0.8,
#     labelspacing=0.1,
#     borderpad=0.1,
# )
# ax.grid(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.set_xlim(-10, 24)
ax.set_ylim(-12, 24)

ax.set_xticks([])
ax.set_yticks([])

text_pos = {
    "Ti": (-8, 10),
    "Mn": (-3, -1),
    "Fe": (2, 17),
    "Cu": (7, 22),
    "Ni": (17, 16),
    "Co": (19, 7),
    "V": (11, 8.5),
    "Cr": (5, 7),
}
for d, pos in text_pos.items():
    ax.text(
        pos[0],
        pos[1],
        d,
        fontsize=FONTSIZE * 0.8,
        # color=colors[c],
        fontweight="bold",
        # put backgroud color
        # bbox=dict(facecolor="white", alpha=0.5, edgecolor=colors[c]),
        bbox=dict(facecolor="white", alpha=0.5, edgecolor=colors[d]),
    )
ax.set_title(
    "UMAP Projection",
    fontsize=FONTSIZE * 1.1,
    pad=20,
    fontweight="bold",
    loc="left",
    # location below top
    y=0.9,
    x=0.04,
)

plt.tight_layout()
plt.savefig("umap_projection.pdf", dpi=300, bbox_inches="tight")


# %%

np.save("umap_proj.npy", umap_proj)
np.save("umap_plt_colors.npy", umap_plt_colors)
# umap_proj = np.load("umap_proj.npy")
# umap_plt_colors = np.load("umap_plt_colors.npy")

# %%
