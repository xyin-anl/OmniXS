# %%

# %load_ext autoreload
# %autoreload 2
from time import time

from dscribe.descriptors import ACSF, LMBTR, SOAP
from src.data.dscribe_featurizer import DscribeFeaturizer
from src.data.ml_data import FeatureProcessor
from src.data.ml_data import DataQuery, load_xas_ml_data

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
from src.data.ml_data import DataQuery, XASPlData, load_xas_ml_data
from src.models.trained_models import LinReg, MeanModel, Trained_FCModel
from utils.src.lightning.pl_module import PLModule
from utils.src.misc.icecream import ic
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.torch.simple_torch_models import SimpleTorchFCModel

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

# ==============================================================================
# PLOT top predictions for FCModel and LinReg
# ==============================================================================

model_class = Trained_FCModel
# model_class = LinReg
splits = 10
simulation_type = "ACSF"
fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(20, 20))
# fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(12,6))
for i, compound in enumerate(cfg.compounds):
    query = DataQuery(compound, simulation_type)
    model = model_class(query)
    Plot().plot_top_predictions(
        model.top_predictions(splits=splits),
        compound,
        splits=splits,
        fill=True,
        axs=axs[:, i],
    )
plt.suptitle(
    f"Top {splits} predictions for {model_class.__name__}", fontsize=24, y=1.02
)
plt.tight_layout()
Plot().save(f"top_{splits}_predictions_{model_class.__name__}")


# ==============================================================================
# ==============================================================================
# PLOT MSE FOR ALL COMPOUNDS for FCModel and LinReg
# ==============================================================================

# fc_models = [
#     Trained_FCModel(DataQuery(compound, "FEFF")) for compound in cfg.compounds
# ]

lin_models = [LinReg(DataQuery(compound, "FEFF")) for compound in cfg.compounds]
fc_acsf_models = [
    Trained_FCModel(DataQuery(compound, "ACSF")) for compound in cfg.compounds
]
from src.analysis.plots import Plot

fig = plt.figure(figsize=(10, 8))
# Plot().bar_plot_of_loss(fc_models, compare_with_mean_model=True)
Plot().bar_plot_of_loss(lin_models, compare_with_mean_model=True)
Plot().bar_plot_of_loss(fc_acsf_models, compare_with_mean_model=True)
# ==============================================================================

# %%

simulation_type = "ACSF"

## generating dsribe features for all compounds
{
    c: load_xas_ml_data(
        DataQuery(c, simulation_type),
        reduce_dims=False,
    ).train.X.shape[1]
    for c in cfg.compounds
}

# %%
pre_reduced_dims["Cu"].train.tensors[0].shape
# # caching pca,scaler
# for c in tqdm(cfg.compounds):
#     load_xas_ml_data(DataQuery(c, simulation_type), reduce_dims=True)


# %%

# =============================================================================
# PLOT PCA EXPLAINED VARIANCE for all compounds for ACSF and SOAP
# =============================================================================
for simulation_type in ["ACSF", "SOAP", "FEFF"]:
    plt.style.use(["default", "science"])
    fig = plt.figure(figsize=(10, 8))
    pca_dims = {
        c: FeatureProcessor(DataQuery(c, simulation_type)).pca.n_components_
        for c in cfg.compounds
    }
    for c in np.array(cfg.compounds)[np.argsort(list(pca_dims.values()))]:
        pca = FeatureProcessor(DataQuery(c, simulation_type)).pca
        label = f"{c}: {pca.n_components_}"
        label += f" ({pca.explained_variance_ratio_.sum():.2f})"
        kwargs = {"label": label, "marker": "o"}
        x = np.arange(1, pca.n_components_ + 1)
        y = pca.explained_variance_ratio_
        y = np.cumsum(y)
        plt.plot(x, y, **kwargs)
    # title
    pre_reduced_dims = [
        load_xas_ml_data(
            DataQuery(c, simulation_type), reduce_dims=False
        ).train.X.shape[1]
        for c in cfg.compounds
    ]
    assert len(set(pre_reduced_dims)) == 1
    title = f"{(simulation_type if simulation_type != 'FEFF' else 'm3gnet')}"
    title += ": Cumulative Explained Variance for PCA"
    title += f"\n Original dimension {set(pre_reduced_dims)}"
    plt.title(title, fontsize=18)
    # misc
    plt.xlabel("Principal Component Index", fontsize=20)
    plt.ylabel("Cumulative Explained Variance", fontsize=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(
        title="Compound: dims (total_explained_variance)",
        fontsize=18,
        title_fontsize=18,
    )
    plt.savefig(
        f"pca_explained_variance_{simulation_type}.pdf",
        bbox_inches="tight",
        dpi=300,
    )
# =============================================================================


# %%
