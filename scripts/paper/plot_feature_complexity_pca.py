# %%
import matplotlib.lines as mlines
import matplotlib as mpl
from src.models.trained_models import (
    MeanModel,
    LinReg,
    Trained_FCModel,
    XGBReg,
    ElastNet,
)
from scipy.stats import gaussian_kde
from config.defaults import cfg
import scienceplots
from matplotlib import pyplot as plt
import random
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from scripts.plots.plot_all_spectras import MLDATAPlotter
from src.data.ml_data import load_all_data
from src.models.trained_models import Trained_FCModel
from src.data.ml_data import DataQuery, load_xas_ml_data
import numpy as np
from src.data.ml_data import FeatureProcessor
from src.models.trained_models import Trained_FCModel, MeanModel, LinReg
from p_tqdm import p_map

# %%

simulation_types = ["ACSF", "SOAP", "FEFF"]


def get_pca_dims(simulation_type):
    out = p_map(
        lambda c: load_xas_ml_data(DataQuery(c, simulation_type)).train.X.shape[1],
        cfg.compounds,
        desc=f"PCA dims {simulation_type}",
    )
    out = {c: out[i] for i, c in enumerate(cfg.compounds)}
    return out


pca_dims_of_simulations = {
    simulation_type: get_pca_dims(simulation_type)
    for simulation_type in simulation_types
}

# %%


def get_mse_relative_to_mean_model(simulation_type, model_class):
    out = p_map(
        lambda c: model_class(DataQuery(c, simulation_type)).mse_relative_to_mean_model,
        cfg.compounds,
        desc=f"{model_class.__name__} {simulation_type}",
    )
    out = {c: out[i] for i, c in enumerate(cfg.compounds)}
    return out


mlp_losses = {
    simulation_type: get_mse_relative_to_mean_model(simulation_type, XGBReg)
    for simulation_type in simulation_types
}
linreg_losses = {
    simulation_type: get_mse_relative_to_mean_model(simulation_type, LinReg)
    for simulation_type in simulation_types
}


# mlp_losses = {
#     simulation_type: {
#         c: ElastNet(
#             DataQuery(c, simulation_type), name="per_compound_tl"
#         ).mse_relative_to_mean_model
#         for c in cfg.compounds
#     }
#     for simulation_type in simulation_types
# }
# linreg_losses = {
#     simulation_type: {
#         c: XGBReg(
#             DataQuery(c, simulation_type), name="per_compound_tl"
#         ).mse_relative_to_mean_model
#         for c in cfg.compounds
#     }
#     for simulation_type in simulation_types
# }


# convert to rmse
for simulation_type in simulation_types:
    for c in cfg.compounds:
        linreg_losses[simulation_type][c] = np.sqrt(linreg_losses[simulation_type][c])
        mlp_losses[simulation_type][c] = np.sqrt(mlp_losses[simulation_type][c])


# %%

ASPECT_RATIO = 4 / 3
HEIGHT = 10
WEIGHT = HEIGHT / ASPECT_RATIO
DPI = 300
FONTSIZE = 14
plt.style.use(["default", "science", "tableau-colorblind10"])
mpl.rcParams["font.size"] = FONTSIZE
mpl.rcParams["figure.dpi"] = DPI
mpl.rcParams["figure.figsize"] = (WEIGHT, HEIGHT)
mpl.rcParams["savefig.dpi"] = DPI
mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"

# %%

fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science", "tableau-colorblind10"])
_markers = ["o", "s", "D"]
ax.plot(
    [
        np.mean(list(pca_dims_of_simulations[sim_type].values()))
        for sim_type in simulation_types
    ],
    [np.mean(list(mlp_losses[sim_type].values())) for sim_type in simulation_types],
    label="MLP",
    color="red",
)
ax.plot(
    [
        np.mean(list(pca_dims_of_simulations[sim_type].values()))
        for sim_type in simulation_types
    ],
    [np.mean(list(linreg_losses[sim_type].values())) for sim_type in simulation_types],
    label="LinReg",
    linestyle="--",
    color="blue",
)
# plot different marker for different simulation type
for i, sim_type in enumerate(simulation_types):
    ax.scatter(
        np.mean(list(pca_dims_of_simulations[sim_type].values())),
        np.mean(list(mlp_losses[sim_type].values())),
        label=f"MLP {sim_type}",
        marker=_markers[i],
        s=50,
        color="red",
    )
    ax.scatter(
        np.mean(list(pca_dims_of_simulations[sim_type].values())),
        np.mean(list(linreg_losses[sim_type].values())),
        label=f"LinReg {sim_type}",
        marker=_markers[i],
        s=50,
        color="blue",
        facecolors="none",
    )
# fill area based on std
ax.fill_between(
    [
        np.mean(list(pca_dims_of_simulations[sim_type].values()))
        for sim_type in simulation_types
    ],
    [
        np.mean(list(mlp_losses[sim_type].values()))
        - np.std(list(mlp_losses[sim_type].values()))
        for sim_type in simulation_types
    ],
    [
        np.mean(list(mlp_losses[sim_type].values()))
        + np.std(list(mlp_losses[sim_type].values()))
        for sim_type in simulation_types
    ],
    color="red",
    alpha=0.1,
)
ax.fill_between(
    [
        np.mean(list(pca_dims_of_simulations[sim_type].values()))
        for sim_type in simulation_types
    ],
    [
        np.mean(list(linreg_losses[sim_type].values()))
        - np.std(list(linreg_losses[sim_type].values()))
        for sim_type in simulation_types
    ],
    [
        np.mean(list(linreg_losses[sim_type].values()))
        + np.std(list(linreg_losses[sim_type].values()))
        for sim_type in simulation_types
    ],
    color="blue",
    alpha=0.1,
)


ax.set_ylabel(r"$\langle \mathrm{RMSE_{mean}/RMSE} \rangle$", fontsize=FONTSIZE)

x_value = cfg.dscribe.pca.n_components
ax.set_xlabel(
    rf"$\bar{{d}}_{{X={x_value}}}^{{F}}$",
    fontsize=FONTSIZE,
)

# X Tick
mean_pcas = {
    "ACSF": np.mean(list(pca_dims_of_simulations["ACSF"].values())),
    "SOAP": np.mean(list(pca_dims_of_simulations["SOAP"].values())),
    "M3GNet": np.mean(list(pca_dims_of_simulations["FEFF"].values())),
}
ax.set_xticks(list(mean_pcas.values()))
ax.set_xticklabels(
    [f"{v:.1f}\n(F={k})" for k, v in mean_pcas.items()], fontsize=0.8 * FONTSIZE
)

# Y Tick
ax.set_yticks(np.arange(1, 3.5, 0.5))
ax.set_yticklabels(
    [f"{v:.1f}" for v in np.arange(1, 3.5, 0.5)], fontsize=0.8 * FONTSIZE
)

# Custom legend handles
linreg_acsf = mlines.Line2D(
    [], [], color="blue", marker="o", linestyle="--", markersize=10, label="LinReg ACSF"
)
mlp_acsf = mlines.Line2D(
    [], [], color="red", marker="o", linestyle="-", markersize=10, label="MLP ACSF"
)
linreg_soap = mlines.Line2D(
    [], [], color="blue", marker="s", linestyle="--", markersize=10, label="LinReg SOAP"
)
mlp_soap = mlines.Line2D(
    [], [], color="red", marker="s", linestyle="-", markersize=10, label="MLP SOAP"
)
linreg_feff = mlines.Line2D(
    [],
    [],
    color="blue",
    marker="d",
    linestyle="--",
    markersize=10,
    label="LinReg M3GNet",
)
mlp_feff = mlines.Line2D(
    [], [], color="red", marker="d", linestyle="-", markersize=10, label="MLP M3GNet"
)
plt.legend(
    handles=[linreg_acsf, linreg_soap, linreg_feff, mlp_acsf, mlp_soap, mlp_feff],
    fontsize=FONTSIZE,
    loc="upper left",
    ncol=2,
    columnspacing=1.0,
)

plt.tight_layout()
plt.savefig("feature_complexity_pca.pdf", dpi=300, bbox_inches="tight")

# %%

# fig, ax = plt.subplots(figsize=(10, 8))
# _markers = ["o", "s", "D"]
# _colors = ["r", "b", "g"]
# for i, sim_type in enumerate(simulation_types):
#     pcas = list(pca_dims_of_simulations[sim_type].values())
#     losses = list(mlp_losses[sim_type].values())
#     ax.scatter(
#         pcas,
#         losses,
#         label=f"MLP {sim_type}",
#         marker=_markers[i],
#         s=50,
#         # alpha=0.5,
#         color=_colors[i],
#     )
#     ax.scatter(
#         pcas,
#         list(linreg_losses[sim_type].values()),
#         label=f"LinReg {sim_type}",
#         marker=_markers[i],
#         s=50,
#         # alpha=0.25,
#         color=_colors[i],
#         # color="black",
#         # no face color
#         facecolors="none",
#     )
# ax.plot(
#     [
#         np.mean(list(pca_dims_of_simulations[sim_type].values()))
#         for sim_type in simulation_types
#     ],
#     [np.mean(list(mlp_losses[sim_type].values())) for sim_type in simulation_types],
#     color="black",
#     label="MLP Mean",
# )
# ax.plot(
#     [
#         np.mean(list(pca_dims_of_simulations[sim_type].values()))
#         for sim_type in simulation_types
#     ],
#     [np.mean(list(linreg_losses[sim_type].values())) for sim_type in simulation_types],
#     color="black",
#     linestyle="--",
#     label="LinReg Mean",
# )
# ax.set_xlabel("PCA dims", fontsize=FONTSIZE)
# ax.set_ylabel("RMSE", fontsize=FONTSIZE)
# ax.legend(fontsize=FONTSIZE)
