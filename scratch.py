# %%

# %load_ext autoreload
# %autoreload 2

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
from main import FC_XAS
from src.analysis.plots import Plot
from src.data.feff_data import FEFFData
from src.data.ml_data import XASPlData, load_xas_ml_data
from src.models.trained_models import LinReg, MeanModel, Trained_FCModel
from utils.src.lightning.pl_module import PLModule
from utils.src.misc.icecream import ic
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.torch.simple_torch_models import SimpleTorchFCModel
from src.data.dscribe_featurizer import DscribeFeaturizer


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
# # model_class = Trained_FCModel
# model_class = LinReg
# splits = 10
# fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(20, 20))
# # fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(12,6))
# for i, compound in enumerate(cfg.compounds):
#     query = DataQuery(compound, "FEFF")
#     model = model_class(query)
#     Plot().plot_top_predictions(
#         model.top_predictions(splits=splits),
#         compound,
#         splits=splits,
#         fill=True,
#         axs = axs[:, i]
#     )
# plt.suptitle(f"Top {splits} predictions for {model_class.__name__}", fontsize=24, y=1.02)
# plt.tight_layout()
# Plot().save(f"top_{splits}_predictions_{model_class.__name__}")
# # ==============================================================================
# # ==============================================================================
# # PLOT MSE FOR ALL COMPOUNDS for FCModel and LinReg
# # ==============================================================================
# fc_models = [
#     Trained_FCModel(DataQuery(compound, "FEFF")) for compound in cfg.compounds
# ]
# lin_models = [
#     LinReg(DataQuery(compound, "FEFF")) for compound in cfg.compounds
# ]
# from src.analysis.plots import Plot
# fig = plt.figure(figsize=(10,8))
# Plot().bar_plot_of_loss(fc_models, compare_with_mean_model=True)
# Plot().bar_plot_of_loss(lin_models, compare_with_mean_model=True).save("mse_all")
# # ==============================================================================
# %%

featurizer = DscribeFeaturizer(ACSF, "Cu")