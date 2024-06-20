# %%
from scripts.paper.universal_TL_mses import universal_model_mses
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
ax.set_xticklabels(["Universal", "Expert", "Tuned-universal"], fontsize=FONTSIZE * 0.8)
# ax.set_ylabel("MSE relative to Mean Model", fontsize=FONTSIZE)
ax.set_ylabel(r"Performance over Baseline ($\eta$)", fontsize=FONTSIZE)
ax.legend(fontsize=FONTSIZE * 0.8)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, None)
ax.set_xlabel("Model Adaptation Strategy", fontsize=FONTSIZE, labelpad=10)
plt.tight_layout()
plt.savefig(
    "model_performance_by_transfer_learning_strategy.pdf", bbox_inches="tight", dpi=300
)

# %%
