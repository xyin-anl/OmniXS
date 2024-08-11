# %%
import ast
from copy import deepcopy
from itertools import product

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt
from p_tqdm import p_map

from config.defaults import cfg
from src.data.ml_data import DataQuery
from src.models.trained_models import (
    ElastNet,
    LinReg,
    MeanModel,
    RFReg,
    RidgeReg,
    Trained_FCModel,
    XGBReg,
    SVReg,
)

# %%


# def train_and_save_models(compounds, features, models):
#     # lambda inp: inp[2](query=DataQuery(inp[0], inp[1])).cache_trained_model(),
#     p_map(
#         lambda inp: inp[2](query=DataQuery(inp[0], inp[1])).cache_trained_model(),
#         product(compounds, features, models),
#         desc="Training models",
#         total=len(compounds) * len(features) * len(models),
#     )
# train_and_save_models(
#     compounds=cfg.compounds,
#     features=["ACSF", "SOAP", "FEFF"],
#     models=[SVReg]
#     # models=[LinReg, RidgeReg, ElastNet, RFReg, XGBReg],
# )


# %%


# =============================================================================
# COMPUTE PERFORMANCE FROM SAVED MODELS
# =============================================================================
def evaluate_cached_model_performance(
    compounds, features, models, METRIC="mse_per_spectra"
):

    metrics = p_map(
        lambda inp: getattr(inp[2](query=DataQuery(inp[0], inp[1])).load(), METRIC),
        product(compounds, features, models),
        desc="Calculating performance",
        total=len(compounds) * len(features) * len(models),
    )

    model_names = [m.__name__ for m in models]
    metrics = dict(zip(product(compounds, features, model_names), metrics))
    return metrics


perf = evaluate_cached_model_performance(
    compounds=cfg.compounds,
    features=["ACSF", "SOAP", "FEFF"],
    # models=[LinReg, RidgeReg, ElastNet, RFReg, XGBReg],
    models=[LinReg, XGBReg, RFReg, ElastNet, RidgeReg, SVReg],
)
# =============================================================================

# %%
# =============================================================================
# add for trainedFC model with names: per_compound_tl, universal_tl, ft_tl
# =============================================================================
for compound, features, mlp_model_name in product(
    cfg.compounds,
    ["ACSF", "SOAP", "FEFF"],
    ["per_compound_tl"],
):
    # if features == "FEFF":
    #     mlp_model_name = "ft_tl"
    model = Trained_FCModel(query=DataQuery(compound, features), name=mlp_model_name)
    perf[(compound, features, "MLP")] = model.mse_per_spectra
# ==============================================================================

# %%


# =============================================================================
# CREATE DATAFRAME FOR PLOTTING WITH MULTIPLE PERFORMANCE METRICS
# =============================================================================
df = pd.DataFrame(
    [
        list(key) + [str(value.tolist())]
        for key, value in perf.items()
        if isinstance(value, np.ndarray)
    ],
    columns=["compound", "feature", "model", "mse_per_spectra"],
)

for compound, feature, mlp_model_name in product(
    cfg.compounds,
    ["ACSF", "SOAP", "FEFF"],
    ["LinReg", "XGBReg", "RFReg", "MLP", "ElastNet", "RidgeReg", "SVReg"],
):
    # add column with geometric mean of mse for each compound
    mask = (
        (df["compound"] == compound)
        & (df["feature"] == feature)
        & (df["model"] == mlp_model_name)
    )
    mse_per_spectra = ast.literal_eval(df[mask]["mse_per_spectra"].values[0])

    baseline_model = MeanModel(query=DataQuery(compound, feature))

    df.loc[mask, "gmean"] = np.exp(np.mean(np.log(mse_per_spectra)))
    df.loc[mask, "baseline_gmean"] = baseline_model.geometric_mean_of_mse_per_spectra
    df.loc[mask, "performance_gmean"] = df[mask]["baseline_gmean"] / df[mask]["gmean"]

    df.loc[mask, "mean"] = np.mean(mse_per_spectra)
    df.loc[mask, "baseline_mean"] = np.mean(baseline_model.mse_per_spectra)
    df.loc[mask, "performance_mean"] = df[mask]["baseline_mean"] / df[mask]["mean"]

    # median
    df.loc[mask, "median"] = np.median(mse_per_spectra)
    df.loc[mask, "baseline_median"] = np.median(baseline_model.mse_per_spectra)
    df.loc[mask, "performance_median"] = (
        df[mask]["baseline_median"] / df[mask]["median"]
    )

# %%

# performance_gmean or performance_mean
# PLOT_METRIC = "performance_mean"
# PLOT_METRIC = "performance_gmean"
# PLOT_METRIC = "performance_mse_of_mses"
PLOT_METRIC = "performance_median"


PLOT_MODELS = ["LinReg", "SVReg", "MLP"]
labels = ["Linear Regression", "Support Vector Regression", "MLP"]

WIDTH = 0.15
FONTSIZE = 18


# plot scatter plot of mse for each model
fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
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


def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = mcolors.to_rgba(c)
    c = [1 - (1 - component) * amount for component in c]
    return c


model_colors = {
    k: v for k, v in zip(PLOT_MODELS, plt.get_cmap("tab10")(range(len(PLOT_MODELS))))
}


for i, feature in enumerate(["ACSF", "SOAP", "FEFF"], start=0):
    for j, mlp_model_name in enumerate(PLOT_MODELS, start=-1):

        mask = (df["feature"] == feature) & (df["model"] == mlp_model_name)
        data = df[mask][PLOT_METRIC]

        ax.boxplot(
            data,
            positions=[i + j * WIDTH],
            widths=WIDTH,
            patch_artist=True,
            boxprops=dict(
                facecolor=lighten_color(model_colors[mlp_model_name], amount=0.5)
            ),
            capprops=dict(color="black"),
            whiskerprops=dict(color="gray"),
            flierprops=dict(markeredgecolor="gray"),
            medianprops=dict(color="black"),
            showfliers=True,  # no outlier
        )

        # ax.scatter(
        #     [i + j * WIDTH] * len(data),
        #     data,
        #     color=model_colors[mlp_model_name],
        #     marker=markers_dict[feature],
        #     label=mlp_model_name,
        # )

        # # plot violin plot instead with all mse values (NOT PLOT_METRIC)
        # for i, compound in enumerate(cfg.compounds):
        #     mask = (
        #         (df["feature"] == feature)
        #         & (df["model"] == mlp_model_name)
        #         & (df["compound"] == compound)
        #     )
        #     string_values = df[mask]["mse_per_spectra"].values
        #     float_values = [ast.literal_eval(x) for x in string_values]
        #     float_values = np.array(float_values)
        #     float_values = float_values.T
        #     ax.violinplot(
        #         float_values,
        #         positions=[i + j * WIDTH],
        #         widths=WIDTH,
        #     )

    ax.set_yscale("log")
    ax.set_ylim(1, 23)
    ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.5)
    # ax.yaxis.grid(True, which="minor", linestyle="--", linewidth=0.5)

    ax.set_yticks(np.arange(1, ax.get_ylim()[1], 2))
    ax.set_yticklabels([f"{int(x)}" for x in ax.get_yticks()], fontsize=FONTSIZE * 0.8)
    ax.xaxis.set_ticks_position("none")

    # ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticks([0, 0.95, 1.9])
    ax.set_xticklabels(["ACSF", "SOAP", "Transfer-feature"], fontsize=FONTSIZE * 0.8)

    ax.set_xlim(-2 * WIDTH, 2 + (len(PLOT_MODELS) - 1) * WIDTH)

    # ax.set_title(PLOT_METRIC, fontsize=FONTSIZE)

# handles by color for model
handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        fc=lighten_color(model_colors[model], amount=0.5),
        edgecolor="black",
    )
    for model in PLOT_MODELS
]


ax.legend(handles, labels, fontsize=FONTSIZE * 0.8, loc="upper left")


ax.set_ylabel(r"Performance ($\eta$)", fontsize=FONTSIZE, labelpad=3)
ax.set_xlabel("Features", fontsize=FONTSIZE, labelpad=10)
plt.tight_layout()
# plt.savefig("model_performance_by_feature_boxplot.png", bbox_inches="tight", dpi=300)
plt.savefig("model_performance_by_feature_boxplot.pdf", bbox_inches="tight", dpi=300)

# %%

# # =============================================================================
# # FOR TABLE in the paper
# # =============================================================================


# df_subset = deepcopy(
#     df[(df["model"] == "LinReg") | (df["model"] == "XGBReg") | (df["model"] == "MLP")]
# )

# mask based on PLOT_MODELS
df_subset = deepcopy(df[df["model"].isin(PLOT_MODELS)])

pivot_df = df_subset.pivot(
    index="compound", columns=["feature", "model"], values="performance_gmean"
)
new_order = []
for feature in ["ACSF", "SOAP", "FEFF"]:
    for model in PLOT_MODELS:
        # for model in labels:
        new_order.append((feature, model))
pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_tuples(new_order))
pivot_df = pivot_df.reindex(cfg.compounds)
pivot_df = pivot_df.rename(columns={"FEFF": "M3GNet"}, level=0)


# round to 3 decimal places with no zero padding in end upto 3 decimal places only
def format_float(x):
    if pd.isna(x):
        return ""
    return f"{x:.3f}"  # Always show 3 decimal places


# Apply the custom formatting to the DataFrame
pivot_df = pivot_df.map(format_float)


latex_output = pivot_df.to_latex()
with open("performance_table.tex", "w") as f:
    f.write(latex_output)

print(latex_output)

# # =============================================================================


# %%
# round to 3 decimal places with no zero padding in end upto 3 decimal places only
def format_float(x):
    if pd.isna(x):
        return ""
    return f"{x:.4f}"  # Always show 3 decimal places


# table of baseline gmean
dict = {}
for compound in cfg.compounds:
    dict[compound] = MeanModel(
        query=DataQuery(compound, "FEFF")
    ).geometric_mean_of_mse_per_spectra
for compound in ["Ti", "Cu"]:
    dict[compound + "(VASP)"] = MeanModel(
        query=DataQuery(compound, "VASP")
    ).geometric_mean_of_mse_per_spectra
dict


# now save as latex
df = pd.DataFrame.from_dict(dict, orient="index", columns=["Baseline"])

# round as before
df = df.map(format_float)

latex_output = df.to_latex()
with open("baseline_table.tex", "w") as f:
    f.write(latex_output)

print(latex_output)

# %%

# # sample svg regression plot
# compound = "Ti"
# svgreg_model = SVReg(query=DataQuery(compound, "FEFF")).load()
# # plt.plot(
# #     svgreg_model.predictions.T,
# # )
# mlp_model = Trained_FCModel(query=DataQuery(compound, "FEFF"), name="per_compound_tl")
# plt.hist(np.log(mlp_model.mse_per_spectra), bins=50, alpha=0.5, label="MLP")
# plt.hist(np.log(svgreg_model.mse_per_spectra), bins=50, alpha=0.5, label="SVR")
# plt.legend()

#
