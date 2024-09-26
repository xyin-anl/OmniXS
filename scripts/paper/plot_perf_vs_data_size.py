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
# add for vasp
data_sizes["Cu_VASP"] = load_xas_ml_data(
    DataQuery("Cu", "VASP"), use_cache=True
).train.X.shape[0]
expert_mses["Cu_VASP"] = Trained_FCModel(
    DataQuery("Cu", "VASP"), name="per_compound_tl"
).median_relative_to_mean_model

data_sizes["Ti_VASP"] = load_xas_ml_data(
    DataQuery("Ti", "VASP"), use_cache=True
).train.X.shape[0]
expert_mses["Ti_VASP"] = Trained_FCModel(
    DataQuery("Ti", "VASP"), name="per_compound_tl"
).median_relative_to_mean_model


# %%

fig, ax = plt.subplots(figsize=(8, 7))
plt.style.use(["default", "science"])
FONTSIZE = 20
colors = plt.cm.tab10.colors[: len(data_sizes)]
vasp_data = {c: d for c, d in data_sizes.items() if "VASP" in c}
feff_data = {c: d for c, d in data_sizes.items() if not "VASP" in c}
args = {"s": 500, "alpha": 0.5, "edgecolors": "black"}
ax.scatter(
    feff_data.values(),
    feff_data.values(),
    c=plt.cm.tab10.colors[: len(feff_data)],
    marker="s",
    **args,
)
ax.scatter(
    vasp_data.values(),
    vasp_data.values(),
    c=plt.cm.tab10.colors[len(feff_data) : len(data_sizes)],
    marker="o",
    **args,
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
ax.set_ylim(0, 23)


from matplotlib.patches import Rectangle, Circle
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPatch
from matplotlib import patches as mpatches


for c, sz, mse in zip(data_sizes.keys(), data_sizes.values(), expert_mses.values()):
    if "_VASP" in c:
        color = plt.cm.tab10.colors[len(feff_data) : len(data_sizes)][
            list(vasp_data.keys()).index(c)
        ]
        hatch = "///"
    else:
        color = plt.cm.tab10.colors[: len(feff_data)][list(feff_data.keys()).index(c)]
        hatch = None

    ax.text(
        sz,
        mse,
        c.split("_")[0],
        fontsize=FONTSIZE,
        ha="center",
        va="center",
        color="black",
        bbox=dict(
            boxstyle="square,pad=0.3",
            facecolor=color,
            alpha=0.4,
            edgecolor="black",
            hatch=hatch,
        ),
        zorder=3,  # Ensure text is on top
    )

    # Add a separate patch for the hatch (if applicable)
    if hatch:
        rect = Rectangle(
            (sz - 0.1, mse - 0.1),
            0.2,
            0.2,
            edgecolor="black",
            hatch=hatch,
            alpha=0.5,
            zorder=2,
            fill=False,
        )
        ax.add_patch(rect)


class HandlerSquare(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Rectangle(
            xy=(center[0] - width / 2, center[1] - height / 2),
            width=width,
            height=height,
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


# Create custom legend
legend_elements = [
    Rectangle(
        (0, 0),
        width=1,
        height=1,
        facecolor="lightgray",
        edgecolor="gray",
        label="FEFF",
    ),
    Rectangle(
        (0, 0),
        width=1,
        height=1,
        facecolor="white",
        edgecolor="gray",
        label="VASP",
        # alpha=0.3,
        hatch="///",
    ),
]

# Add the legend with custom handler and larger markers
ax.legend(
    handles=legend_elements,
    loc="best",
    handler_map={mpatches.Rectangle: HandlerSquare()},
    handlelength=3,
    handleheight=3,
)

# increase font size of legend
plt.setp(ax.get_legend().get_texts(), fontsize=FONTSIZE*0.8)

plt.tight_layout()
plt.savefig("performace_vs_data_size.pdf", dpi=300, bbox_inches="tight")
