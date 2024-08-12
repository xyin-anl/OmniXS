# %%
from scipy.optimize import curve_fit

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
from src.models.trained_models import Trained_FCModel, MeanModel

# %%

data_sizes = {}
train_variances = {}
for c in cfg.compounds:
    ml_split = load_xas_ml_data(DataQuery(c, "FEFF"), use_cache=True)
    data_sizes[c] = (
        # ml_split.train.X.shape[0] + ml_split.val.X.shape[0] + ml_split.test.X.shape[0]
        ml_split.train.X.shape[0]
    )
    train_variances[c] = ml_split.train.y.var(axis=0).mean()

expert_mses = {
    c: Trained_FCModel(
        DataQuery(c, "FEFF"), name="per_compound_tl"
    ).median_relative_to_mean_model
    for c in cfg.compounds
}

# %%

fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
FONTSIZE = 18
colors = plt.cm.tab10.colors[:8]
ax.scatter(
    data_sizes.values(),
    expert_mses.values(),
    s=500,
    # size based on variance
    # s=5e4 * np.array(list(train_variances.values())),
    c=colors,
    # use square marker
    marker="s",
    alpha=0.5,
    # use edge color
    edgecolors="black",
)
ax.set_xlabel("Data size", fontsize=FONTSIZE)
ax.set_ylabel(r"Performance ($\eta$)", fontsize=FONTSIZE)

ax.set_xticklabels(ax.get_xticks(), fontsize=0.8 * FONTSIZE)
ax.get_xaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)))
)

ax.set_ylim(0, 25)
ax.set_yticks([0, 5, 10, 15, 20, 25])
# integer
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x)))
ax.set_yticklabels(ax.get_yticks(), fontsize=0.8 * FONTSIZE)

# no minor ticks
ax.minorticks_off()
# horizing gird only for y axis with very small alpha
ax.grid(axis="y", alpha=0.4, linestyle="--")

# ax.set_ylim(1, None)

# put compund name in center and color it by darker version of the color
for c, (sz, mse) in zip(cfg.compounds, zip(data_sizes.values(), expert_mses.values())):
    ax.text(
        sz,
        mse,
        c,
        fontsize=FONTSIZE,
        ha="center",
        va="center",
        color="black",
        # bbox=dict(facecolor="white", alpha=0.5, edgecolor="white"),
    )

# ax.set_ylim(3, None)


# # Data: sizes and corresponding expert model MSEs
# x = np.array(list(data_sizes.values()))  # Assuming data_sizes is a dictionary
# y = np.array(list(expert_mses.values()))  # Assuming expert_mses is a dictionary
# sort_idx = np.argsort(x)
# x = x[sort_idx]
# y = y[sort_idx]
# fit = np.poly1d(np.polyfit(x, y, 1))
# y_pred = fit(x)
# ss_res = np.sum((y - y_pred) ** 2)
# ss_tot = np.sum((y - np.mean(y)) ** 2)
# R2 = 1 - (ss_res / ss_tot)
# ax.plot(
#     x,
#     fit(x),
#     color="black",
#     linestyle="--",
#     label=r"R2 = {:.2f}".format(R2),
# )
# ax.legend(fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig("performace_vs_data_size.pdf", dpi=300, bbox_inches="tight")

# %%
