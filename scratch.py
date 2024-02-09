# %%

# %load_ext autoreload
# %autoreload 2
from time import time

import numpy as np
import pytorch_lightning as pl
import torch
from dscribe.descriptors import (
    ACSF,
    LMBTR,
    MBTR,
    SOAP,
    CoulombMatrix,
    EwaldSumMatrix,
    SineMatrix,
)
from lightning import Trainer
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from config.defaults import cfg
from main import FC_XAS
from src.analysis.plots import Plot
from src.data.dscribe_featurizer import DscribeFeaturizer
from src.data.feff_data import FEFFData
from src.data.ml_data import DataQuery, FeatureProcessor, XASPlData, load_xas_ml_data
from src.models.trained_models import LinReg, MeanModel, Trained_FCModel
from utils.src.lightning.pl_module import PLModule
from utils.src.misc.icecream import ic
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.torch.simple_torch_models import SimpleTorchFCModel

# %%

# %%

# ==============================================================================
# PEAK LOCATIONS
# ==============================================================================
# compound = "Cu"
# model_class = Trained_FCModel
# # model_class = LinReg
# model = model_class(DataQuery(compound, "FEFF"))
# fig, axs = plt.subplots(2,4, figsize=(12,8))
# r2_scores = []
# for compound, ax in zip(cfg.compounds, axs.flatten()):
#     model = model_class(DataQuery(compound, "FEFF"))
#     r2 = Plot().plot_peak_loc(model, compound, ax=ax)
#     r2_scores.append(r2)
# for ax in axs[:,0]:
#     ax.set_ylabel("Predicted Peak Location")
# for ax in axs[-1]:
#     ax.set_xlabel("Ground Truth Peak Location")
# plt.suptitle(f"Peak Location for {model_class.__name__}", fontsize=24, y=0.95)
# plt.tight_layout()
# Plot().save(f"peak_loc_{model_class.__name__}")
# # ==============================================================================

# %%

# # ==============================================================================
# # PLOT top predictions for FCModel and LinReg
# # ==============================================================================
# model_class = Trained_FCModel
# # model_class = LinReg
# splits = 10
# simulation_type = "ACSF"
# fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(20, 20))
# # fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(12,6))
# for i, compound in enumerate(cfg.compounds):
#     query = DataQuery(compound, simulation_type)
#     model = model_class(query)
#     Plot().plot_top_predictions(
#         model.top_predictions(splits=splits),
#         compound,
#         splits=splits,
#         fill=True,
#         axs=axs[:, i],
#     )
# plt.suptitle(
#     f"Top {splits} predictions for {model_class.__name__}", fontsize=24, y=1.02
# )
# plt.tight_layout()
# Plot().save(f"top_{splits}_predictions_{model_class.__name__}")


# # ==============================================================================
# # ==============================================================================
# # PLOT MSE FOR ALL COMPOUNDS for FCModel and LinReg
# # ==============================================================================
# # fc_models = [
# #     Trained_FCModel(DataQuery(compound, "FEFF")) for compound in cfg.compounds
# # ]
# lin_models = [LinReg(DataQuery(compound, "FEFF")) for compound in cfg.compounds]
# fc_acsf_models = [
#     Trained_FCModel(DataQuery(compound, "ACSF")) for compound in cfg.compounds
# ]
# fig = plt.figure(figsize=(10, 8))
# # Plot().bar_plot_of_loss(fc_models, compare_with_mean_model=True)
# Plot().bar_plot_of_loss(lin_models, compare_with_mean_model=True)
# Plot().bar_plot_of_loss(fc_acsf_models, compare_with_mean_model=True)
# # ==============================================================================

# %%
# =============================================================================
# caching pca,scaler
# =============================================================================
# simulation_type = "ACSF"
# for c in tqdm(cfg.compounds):
#     load_xas_ml_data(DataQuery(c, simulation_type), reduce_dims=True)
# # =============================================================================


# %%

# # =============================================================================
# # PLOT PCA EXPLAINED VARIANCE for all compounds for ACSF and SOAP
# # =============================================================================
# for simulation_type in ["ACSF", "SOAP", "FEFF"]:
#     plt.style.use(["default", "science"])
#     fig = plt.figure(figsize=(10, 8))
#     pca_dims = {
#         c: FeatureProcessor(DataQuery(c, simulation_type)).pca.n_components_
#         for c in cfg.compounds
#     }
#     for c in np.array(cfg.compounds)[np.argsort(list(pca_dims.values()))]:
#         pca = FeatureProcessor(DataQuery(c, simulation_type)).pca
#         label = f"{c}: {pca.n_components_}"
#         label += f" ({pca.explained_variance_ratio_.sum():.2f})"
#         kwargs = {"label": label, "marker": "o"}
#         x = np.arange(1, pca.n_components_ + 1)
#         y = pca.explained_variance_ratio_
#         y = np.cumsum(y)
#         plt.plot(x, y, **kwargs)
#     # title
#     pre_reduced_dims = [
#         load_xas_ml_data(
#             DataQuery(c, simulation_type), reduce_dims=False
#         ).train.X.shape[1]
#         for c in cfg.compounds
#     ]
#     assert len(set(pre_reduced_dims)) == 1
#     title = f"{(simulation_type if simulation_type != 'FEFF' else 'm3gnet')}"
#     title += ": Cumulative Explained Variance for PCA"
#     title += f"\n Original dimension {set(pre_reduced_dims)}"
#     plt.title(title, fontsize=18)
#     # misc
#     plt.xlabel("Principal Component Index", fontsize=20)
#     plt.ylabel("Cumulative Explained Variance", fontsize=20)
#     plt.yticks(fontsize=18)
#     plt.xticks(fontsize=18)
#     plt.legend(
#         title="Compound: dims (total_explained_variance)",
#         fontsize=18,
#         title_fontsize=18,
#     )
#     plt.savefig(
#         f"pca_explained_variance_{simulation_type}.pdf",
#         bbox_inches="tight",
#         dpi=300,
#     )
# # =============================================================================


# %%

# # =============================================================================
# # PLOT PCA DIMS for all compounds and all simulations
# # =============================================================================
# plt.style.use(["default", "science"])
# plt.figure(figsize=(10, 8))
# data = {
#     c: [
#         FeatureProcessor(DataQuery(c, st)).pca.n_components_
#         for st in ["ACSF", "SOAP", "FEFF"]
#     ]
#     for c in cfg.compounds
# }
# fig, axs = plt.subplots(layout="constrained", figsize=(10, 8))
# for compound, pca_dims in data.items():
#     axs.bar(
#         [
#             f"{compound}_{(st if st!='FEFF' else 'm3gnet')}"
#             for st in ["ACSF", "SOAP", "FEFF"]
#         ],
#         pca_dims,
#         label=compound,
#     )
# plt.xticks(rotation=90, fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(fontsize=16)
# plt.title("PCA dims for ACSF, SOAP and m3gnet", fontsize=20)
# plt.ylabel("PCA dims", fontsize=20)
# plt.savefig("pca_dims.pdf", bbox_inches="tight", dpi=300)
# # =============================================================================

# %%

sim_types = ["ACSF", "SOAP", "FEFF"]
fc_models = {
    c: {sim_type: Trained_FCModel(DataQuery(c, sim_type)) for sim_type in sim_types}
    for c in cfg.compounds
}
linreg_models = {
    c: {sim_type: LinReg(DataQuery(c, sim_type)) for sim_type in sim_types}
    for c in cfg.compounds
}

# %%

fc_mse = {
    c: {sim_type: fc_models[c][sim_type].mse for sim_type in sim_types}
    for c in cfg.compounds
}
linreg_mse = {
    c: {sim_type: linreg_models[c][sim_type].mse for sim_type in sim_types}
    for c in cfg.compounds
}


# %%

# grouped bar plot by compound with linreg showing color same as fcmodel but lighter
# so 6 bars for each compound
# Bar positions

plt.style.use(["default", "science"])
fig, ax = plt.subplots(figsize=(10, 7))

# Example compounds and simulation types, replace with your actual data
compounds = cfg.compounds
n_groups = len(compounds)
n_sim_types = len(sim_types)
total_width = 0.8
single_width = total_width / n_sim_types
space_width = 0
bar_width = single_width - (space_width / n_sim_types)
# Starting position for the first group of bars
start_pos = np.arange(n_groups) - (total_width / 2) + (bar_width / 2)
for i, sim_type in enumerate(sim_types):
    # Positions of the fc and linreg bars for this simulation type
    linreg_pos = start_pos + i * (bar_width + space_width)
    fc_pos = start_pos + i * (bar_width + space_width)
    # Extracting MSE values for fc and linreg
    fc_values = [fc_mse[compound][sim_type] for compound in compounds]
    linreg_values = [linreg_mse[compound][sim_type] for compound in compounds]
    # Plotting the bars
    ax.bar(
        linreg_pos,
        linreg_values,
        bar_width,
        # label=f"LinReg MSE: {sim_type}",
        hatch="||",
        # alpha=0.2,
        edgecolor="black",
        fill=False,
        linewidth=1.5,
    )
    ax.bar(
        fc_pos,
        fc_values,
        bar_width,
        label=f"FC MSE: {sim_type if sim_type!='FEFF' else 'm3gnet'}",
        # edgecolor="black",
    )
ax.set_xlabel("Compound", fontsize=20)
ax.set_ylabel("MSE", fontsize=20)
ax.set_title("MSE by compound and simulation type", fontsize=24)
ax.set_xticks(np.arange(n_groups))
ax.set_xticklabels(compounds, fontsize=18)
# add another bar to legend for linreg using patch

# ax.legend(fontsize=18, loc="upper center")

from matplotlib import patches as mpatches

dummy_patch = mpatches.Patch(fill=False, edgecolor="black", hatch="||", label="LinReg")
handles, labels = ax.get_legend_handles_labels()
handles.append(dummy_patch)
labels.append("LinReg")
ax.legend(handles=handles, labels=labels, fontsize=18)
plt.tight_layout()
plt.savefig("mse_by_compound_and_simulation_type.pdf", bbox_inches="tight", dpi=300)

# %%

# mse_diff = {c: fc_mse[c]["FEFF"] - linreg_mse[c]["FEFF"] for c in cfg.compounds}
mse_diff = {c: linreg_mse[c]["FEFF"] - fc_mse[c]["FEFF"] for c in cfg.compounds}


def data_size(ml_splits):
    return sum(
        [split.X.shape[0] for split in [ml_splits.train, ml_splits.val, ml_splits.test]]
    )


data_sizes = {
    c: data_size(load_xas_ml_data(DataQuery(c, "FEFF"))) for c in cfg.compounds
}

# %%

plt.style.use(["default", "science"])
fig, ax = plt.subplots(figsize=(9, 7))
# sort mse_diff by data size
x = np.array(list(data_sizes.values()))
y = np.array(list(mse_diff.values()))
sort_idx = x.argsort()
x = x[sort_idx]
y = y[sort_idx]
ax.plot(x, y, label="MSE difference", markersize=10, linestyle="-", marker="o")
# add point labels for each compound
sorted_compounds = np.array(list(mse_diff.keys()))[sort_idx]
for i, txt in enumerate(sorted_compounds):
    ax.annotate(
        txt,
        (x[i], y[i]),
        fontsize=20,
        textcoords="offset points",
        xytext=(0, 10),
        color="red",
        fontweight="bold",
    )
ax.set_xlabel("Data size", fontsize=20)
ax.set_ylabel("MSE_LinReg - MSE_FC", fontsize=20)
ax.set_title("Performance difference by data size in m3gnet featurization", fontsize=24)
ax.tick_params(axis="both", which="major", labelsize=18)
plt.tight_layout()
plt.savefig("mse_diff_by_data_size.pdf", bbox_inches="tight", dpi=300)

# %%
