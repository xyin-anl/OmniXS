# %%
from src.data.ml_data import load_all_data
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import scienceplots

# draw strip plot
import seaborn as sns
from matplotlib import pyplot as plt
from p_tqdm import p_map
from scipy.stats import gaussian_kde

from config.defaults import cfg
from scripts.paper.universal_TL_mses import universal_model_mses
from scripts.paper.plot_utils import Plot
from src.data.ml_data import DataQuery, DataSplit, MLSplits, load_xas_ml_data
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

all_mses_universal = Trained_FCModel(
    DataQuery("ALL", "FEFF"), name="universal_tl"
).mse_per_spectra

all_mses_expert = np.concatenate(
    [
        Trained_FCModel(DataQuery(c, "FEFF"), name="per_compound_tl").mse_per_spectra
        for c in cfg.compounds
    ]
)
all_mses_ft_tl = np.concatenate(
    [
        Trained_FCModel(DataQuery(c, "FEFF"), name="ft_tl").mse_per_spectra
        for c in cfg.compounds
    ]
)

# %%

mses_univ_per_compound = universal_model_mses(relative_to_per_compound_mean_model=True)[
    "per_compound"
]
mses_per_compound_tl = {
    c: Trained_FCModel(
        DataQuery(c, "FEFF"), name="per_compound_tl"
    ).mse_relative_to_mean_model
    for c in cfg.compounds
}
mses_ft_tl = {
    c: Trained_FCModel(DataQuery(c, "FEFF"), name="ft_tl").mse_relative_to_mean_model
    for c in cfg.compounds
}

# %%
# box plot of mse for each model
FONTSIZE = 18
fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
cmap = "tab10"
colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds)))
violins = ax.violinplot(
    [
        list(mses_univ_per_compound.values()),
        list(mses_per_compound_tl.values()),
        list(mses_ft_tl.values()),
    ],
    positions=[0, 1, 2],
    widths=0.4,
    showmeans=False,
    showextrema=False,
    points=1000,
    bw_method=0.5,
)
for violin, color in zip(violins["bodies"], colors):
    violin.set_facecolor(color)
    violin.set_alpha(0.3)  # Optionally, you can set the alpha for transparency


colors = ["tab:blue", "tab:orange", "tab:green"]
labels = ["Universal model", "Expert model", "Tuned-universal model"]
for i, mses in enumerate([mses_univ_per_compound, mses_per_compound_tl, mses_ft_tl]):
    ax.scatter(
        np.ones(len(mses)) * i,
        list(mses.values()),
        label=labels[i],
        zorder=1,
    )
# horizontal grid
ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5, zorder=0, alpha=0.3)
ax.set_xticks([0, 1, 2])
ax.set_yticklabels([f"{int(x)}" for x in ax.get_yticks()], fontsize=FONTSIZE * 0.8)
FONTSIZE = 18
ax.set_xticklabels(["Universal", "Expert", "Tuned-universal"], fontsize=FONTSIZE * 0.9)
# ax.set_ylabel("MSE relative to Mean Model", fontsize=FONTSIZE)
ax.set_ylabel(r"Performance over Baseline ($\eta$)", fontsize=FONTSIZE)
ax.legend(fontsize=FONTSIZE * 0.8)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, None)
ax.set_xlabel("M3XAS Variants", fontsize=FONTSIZE, labelpad=10)
plt.tight_layout()
plt.savefig(
    "model_performance_by_transfer_learning_strategy.pdf", bbox_inches="tight", dpi=300
)

# %%


import torch

train_X = np.concatenate(
    [load_xas_ml_data(DataQuery(c, "FEFF")).train.X for c in cfg.compounds]
)
train_y = np.concatenate(
    [load_xas_ml_data(DataQuery(c, "FEFF")).train.y for c in cfg.compounds]
)
val_X = np.concatenate(
    [load_xas_ml_data(DataQuery(c, "FEFF")).val.X for c in cfg.compounds]
)
val_y = np.concatenate(
    [load_xas_ml_data(DataQuery(c, "FEFF")).val.y for c in cfg.compounds]
)
test_X = np.concatenate(
    [load_xas_ml_data(DataQuery(c, "FEFF")).test.X for c in cfg.compounds]
)
test_y = np.concatenate(
    [load_xas_ml_data(DataQuery(c, "FEFF")).test.y for c in cfg.compounds]
)

train_data = DataSplit(train_X, train_y)
val_data = DataSplit(val_X, val_y)
test_data = DataSplit(test_X, test_y)
data = MLSplits(train_data, val_data, test_data)

# %%

# KDE of MSE and CDF of MSE

mses_all_universal = Trained_FCModel(
    DataQuery("ALL", "FEFF"), name="universal_tl"
).model(torch.tensor(data.test.X)) - torch.tensor(data.test.y)
mses_all_universal = mses_all_universal**2
mses_all_universal = mses_all_universal.mean(dim=1).detach().numpy()
mses_all_universal


mses_all_expert = np.concatenate(
    [
        Trained_FCModel(DataQuery(c, "FEFF"), name="per_compound_tl").mse_per_spectra
        for c in cfg.compounds
    ]
)

mses_all_ft_tl = np.concatenate(
    [
        Trained_FCModel(DataQuery(c, "FEFF"), name="ft_tl").mse_per_spectra
        for c in cfg.compounds
    ]
)
print(f"Universal: {mses_all_universal.mean()}")
print(f"Expert: {mses_all_expert.mean()}")
print(f"FT-TL: {mses_all_ft_tl.mean()}")

# %%

fig = plt.figure(figsize=(6, 8))
plt.style.use(["default", "science"])
FONTSIZE = 18
cmap = "Set1"
colors = plt.get_cmap(cmap)(range(3))
gs = fig.add_gridspec(2, hspace=0.1)
ax = fig.add_subplot(gs[1])
for d, label, c in zip(
    [mses_all_universal, mses_all_expert, mses_all_ft_tl],
    ["Universal", "Expert", "Tuned-universal"],
    colors,
):
    ax.hist(
        np.log10(d),
        histtype="step",
        cumulative=True,
        density=True,
        bins=100,
        label=label,
        color=c,
    )
ax.set_xlabel(r"$\log_{10}$(MSE)", fontsize=FONTSIZE)
ax.set_ylabel("Cumulative Density", fontsize=FONTSIZE)
ax.tick_params(axis="both", which="major", labelsize=FONTSIZE * 0.8)
ax.set_ylim(0, 1.01)
ax.set_xlim(-4, -1)

# Density plot (smoothed KDE)
ax1 = fig.add_subplot(gs[0], sharex=ax, sharey=ax)
for d, label, color in zip(
    [mses_all_universal, mses_all_expert, mses_all_ft_tl],
    ["Universal", "Expert", "Tuned-universal"],
    colors,
):
    kde = gaussian_kde(np.log10(d))
    x = np.linspace(np.log10(d).min(), np.log10(d).max(), 1000)
    ax1.plot(x, kde(x), label=label, linestyle="-", color=color)
    # ax1.fill_between(x, kde(x), alpha=0.1, label=label)
ax1.set_ylabel("Density", fontsize=FONTSIZE)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.tick_params(axis="both", which="major", labelsize=FONTSIZE * 0.8)
ax1.legend(loc="upper left", fontsize=FONTSIZE * 0.8)
plt.tight_layout()
plt.savefig(
    "model_performance_by_transfer_learning_strategy_cdf.pdf",
    bbox_inches="tight",
    dpi=300,
)

# %%

data, names = load_all_data(return_compound_name=True, sample_method="same_size")
test_data, test_names = data.test, names[2]

# %%
test_X, test_y = test_data.X, test_data.y

# %%
model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl").model


# %%

expert_models = {
    c: Trained_FCModel(DataQuery(c, "FEFF"), name="per_compound_tl").model
    for c in cfg.compounds
}
tuned_univ_models = {
    c: Trained_FCModel(DataQuery(c, "FEFF"), name="ft_tl").model for c in cfg.compounds
}
univ_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl").model
univ_models = {c: univ_model for c in cfg.compounds}

# %%

mses_all = []
for X, y, c in zip(test_X, test_y, test_names):
    mses = []
    for model in [univ_models[c], expert_models[c], tuned_univ_models[c]]:
        mses.append(
            ((model(torch.tensor(X).reshape(1, -1)) - torch.tensor(y)) ** 2)
            .mean()
            .item()
        )
    mses_all.append(mses)
mses_all = np.array(mses_all)

# %%

fig, ax = plt.subplots(figsize=(6, 6))
plt.style.use(["default", "science"])
cmap = "tab10"
colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds)))

for i, c in enumerate(cfg.compounds):
    # expert -> univ
    x = (mses_all[:, 0][test_names == c] - mses_all[:, 1][test_names == c],)

    # univ -> ft-univ
    y = (mses_all[:, 2][test_names == c] - mses_all[:, 0][test_names == c],)

    # # expert -> ft-univ
    # y = (mses_all[:, 2][test_names == c] - mses_all[:, 1][test_names == c],)

    ax.scatter(
        x,
        y,
        s=2,
        c=colors[i],
        label=c,
    )
ax.legend()
ax.set_xlim(-0.02, 0.02)
ax.set_ylim(-0.02, 0.02)
ax.hlines(0, -0.02, 0.02, linestyle="--", color="gray")
ax.vlines(0, -0.02, 0.02, linestyle="--", color="gray")
ax.set_xlabel(r"Expert to Universal")
ax.set_ylabel(r"Universal to Tuned-Universal")
# ax.set_yscale("symlog")

# %%

df = pd.DataFrame(mses_all, columns=["Universal", "Expert", "Tuned-Universal"])
df["Compound"] = test_names

# %%


import pandas as pd
import plotly.graph_objects as go


def create_sankey(df, col1, col2, col3):
    """
    Creates a Sankey diagram based on changes between three columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    col1 (str): The name of the first column.
    col2 (str): The name of the second column.
    col3 (str): The name of the third column.
    """
    # Create partitions
    df["First_to_Second"] = df.apply(
        lambda x: "Increased" if x[col2] > x[col1] else "Decreased", axis=1
    )
    df["Second_to_Third"] = df.apply(
        lambda x: "Increased" if x[col3] > x[col2] else "Decreased", axis=1
    )

    # Define the labels and colors
    labels = [col1, "Increase", "Decrease", "Increase", "Decrease", col3]
    colors = ["blue", "red", "green", "red", "green", "blue"]

    # Count occurrences for Sankey
    source = []
    target = []
    value = []

    # col1 to First_to_Second
    for partition in df["First_to_Second"].unique():
        count = df[df["First_to_Second"] == partition].shape[0]
        source.append(0)  # col1
        target.append(1 if partition == "Increased" else 2)  # Increased or Decreased
        value.append(count)

    # First_to_Second to Second_to_Third
    for partition1 in df["First_to_Second"].unique():
        for partition2 in df["Second_to_Third"].unique():
            count = df[
                (df["First_to_Second"] == partition1)
                & (df["Second_to_Third"] == partition2)
            ].shape[0]
            source.append(1 if partition1 == "Increased" else 2)
            target.append(3 if partition2 == "Increased" else 4)
            value.append(count)

    # Second_to_Third to col3
    final_counts = df["Second_to_Third"].value_counts()
    for partition in final_counts.index:
        source.append(3 if partition == "Increased" else 4)
        target.append(5)  # col3
        value.append(final_counts[partition])

    # Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[col1, "Increase", "Decrease", "Increase", "Decrease", col3],
                    color=colors,
                ),
                link=dict(source=source, target=target, value=value, color="lightgrey"),
            )
        ]
    )

    fig.update_layout(
        title_text=f"Sankey Diagram of {col1}, {col2}, and {col3} Changes",
        font_size=10,
        annotations=[
            dict(
                text=col1, x=0.01, xref="paper", y=-0.2, yref="paper", showarrow=False
            ),
            dict(
                text="Increase",
                x=0.3,
                xref="paper",
                y=-0.2,
                yref="paper",
                showarrow=False,
            ),
            dict(
                text="Decrease",
                x=0.3,
                xref="paper",
                y=-0.25,
                yref="paper",
                showarrow=False,
            ),
            dict(
                text="Increase",
                x=0.7,
                xref="paper",
                y=-0.2,
                yref="paper",
                showarrow=False,
            ),
            dict(
                text="Decrease",
                x=0.7,
                xref="paper",
                y=-0.25,
                yref="paper",
                showarrow=False,
            ),
            dict(
                text=col3, x=0.99, xref="paper", y=-0.2, yref="paper", showarrow=False
            ),
        ],
        showlegend=True,
        legend=dict(
            traceorder="normal",
            itemsizing="constant",
            orientation="h",
            xanchor="center",
            x=0.5,
            y=-0.3,
        ),
    )
    fig.show()


# Usage example
# df = pd.read_csv('/mnt/data/your_actual_data.csv')
# create_sankey(df, 'Expert', 'Universal', 'Tuned-Universal')
create_sankey(df, "Universal", "Expert", "Tuned-Universal")
# %%

best_model = df[df.columns[:3]].idxmin(axis=1)

df["Best Model"] = best_model
fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
df_subset = df.groupby("Compound")["Best Model"].value_counts().unstack()
# put tuneed universal first , then expert, then universal
df_subset = df_subset[["Tuned-Universal", "Expert", "Universal"]]
# plit by pctentage
df_subset = df_subset.div(df_subset.sum(axis=1), axis=0)
df_subset.plot.bar(stacked=True, ax=ax)
FONTSIZE = 18
ax.set_ylabel("Best model shares of spectras", fontsize=FONTSIZE)
ax.set_xlabel("Compound", fontsize=FONTSIZE)
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=FONTSIZE * 0.8)
ax.set_yticklabels([f"{x:.1f}" for x in ax.get_yticks()], fontsize=FONTSIZE * 0.8)
plt.minorticks_off()
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=3,
    fontsize=FONTSIZE * 0.8,
    # title="Model",
    # title_fontsize=FONTSIZE,
)
plt.tight_layout()
fig.savefig(
    "best_model_shares_of_spectras_by_transfer_learning_strategy.pdf",
    bbox_inches="tight",
    dpi=300,
)

# %%
