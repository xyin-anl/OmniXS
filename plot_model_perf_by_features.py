# %%
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt
from p_tqdm import p_map

from config.defaults import cfg
from src.analysis.plots import Plot
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import (
    ElastNet,
    LinReg,
    MeanModel,
    RFReg,
    RidgeReg,
    Trained_FCModel,
    XGBReg,
)

# %%
# =============================================================================
# # COMPUTE PERFORMANCE
# =============================================================================
# def performance(compounds, features, models):
#     out = p_map(
#         lambda inp: inp[2](query=DataQuery(inp[0], inp[1])).mse_relative_to_mean_model,
#         product(compounds, features, models),
#         desc="Calculating performance",
#     )
#     model_names = [m.__name__ for m in models]
#     out = dict(zip(product(compounds, features, model_names), out))
#     return out
# perf = performance(
#     compounds=cfg.compounds,
#     features=["ACSF", "SOAP", "FEFF"],
#     models=[LinReg, RidgeReg, ElastNet, RFReg, XGBReg],
# )
# df = pd.DataFrame(perf, index=["MSE"]).T
# df.to_csv("model_performance.csv")
# =============================================================================
# %%

# # =============================================================================
# # FOR TABLE in the paper
# # =============================================================================
# df = pd.read_csv("dataset/model_performance.csv", index_col=[0])
# df.columns = ["compound", "feature", "model", "normalized_mse"]
# df_subset = deepcopy(
#     df[(df["model"] == "LinReg") | (df["model"] == "XGBReg") | (df["model"] == "MLP")]
# )
# pivot_df = df_subset.pivot(
#     index="compound", columns=["feature", "model"], values="normalized_mse"
# )
# new_order = []
# for feature in ["ACSF", "SOAP", "FEFF"]:
#     for model in ["LinReg", "XGBReg", "MLP"]:
#         new_order.append((feature, model))
# pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_tuples(new_order))
# pivot_df = pivot_df.reindex(cfg.compounds)
# pivot_df = pivot_df.rename(columns={"FEFF": "M3GNet"}, level=0)
# pivot_df.to_csv("model_performance_by_feature.csv")
# latex_output = pivot_df.to_latex()
# with open("performance_table.tex", "w") as f:
#     f.write(latex_output)
# # =============================================================================


# %%

# first row
df = pd.read_csv("dataset/model_performance.csv", index_col=[0])
df.columns = ["compound", "feature", "model", "normalized_mse"]

# plot scatter plot of mse for each model
fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science", "no-latex"])
cmap = "tab10"
colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds)))

markers_dict = {
    "ACSF": "D",
    "SOAP": "s",
    "FEFF": "o",
}

colors_dict = {
    "ACSF": plt.get_cmap("tab10")(0),
    "SOAP": plt.get_cmap("tab10")(1),
    "FEFF": plt.get_cmap("tab10")(2),
}


import matplotlib.colors as mcolors


def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = mcolors.to_rgba(c)
    c = [1 - (1 - component) * amount for component in c]
    return c


FONTSIZE = 18

model_colors = {
    "LinReg": plt.get_cmap("tab10")(0),
    "XGBReg": plt.get_cmap("tab10")(1),
    "MLP": plt.get_cmap("tab10")(2),
}

for i, feature in enumerate(["ACSF", "SOAP", "FEFF"]):

    for j, model in enumerate(["LinReg", "XGBReg", "MLP"]):
        data = df[(df["model"] == model) & (df["feature"] == feature)]["normalized_mse"]
        ax.boxplot(
            data,
            positions=[i * 1.5 + j * 0.2],
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor=lighten_color(model_colors[model], amount=0.5)),
            # capprops=dict(color=model_colors[model]),
            # whiskerprops=dict(color=model_colors[model]),
            # flierprops=dict(markeredgecolor=model_colors[model]),
            # medianprops=dict(color=model_colors[model]),
            capprops=dict(color="black"),
            whiskerprops=dict(color="gray"),
            flierprops=dict(markeredgecolor="gray"),
            medianprops=dict(color="black"),
        )
    ax.set_yscale("log")
    ax.set_ylim(1, 9)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xticks(np.arange(3) * 1.5 + 0.2)
    ax.set_xticklabels(["ACSF", "SOAP", "M3GNet"], fontsize=FONTSIZE * 0.8)
    ax.xaxis.set_ticks_position("none")

# handles by color for model
handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        fc=lighten_color(model_colors[model], amount=0.5),
        edgecolor="black",
    )
    for model in ["LinReg", "XGBReg", "MLP"]
]
labels = ["Linear Regression", "XGBoost Regression", "MLP"]
ax.legend(handles, labels, fontsize=FONTSIZE * 0.8)


ax.set_ylabel("Normalized MSE", fontsize=FONTSIZE)
ax.set_xlabel("Features", fontsize=FONTSIZE)
plt.tight_layout()
plt.savefig("model_performance_by_feature_boxplot.png", bbox_inches="tight", dpi=300)
plt.savefig("model_performance_by_feature_boxplot.pdf", bbox_inches="tight", dpi=300)

# %%
