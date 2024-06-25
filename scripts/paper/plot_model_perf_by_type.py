# %%
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
from src.analysis.plots import Plot
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
