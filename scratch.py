# %%

import pickle
import importlib
import src
from utils.src.misc import icecream
from utils.src.plots.highlight_tick import highlight_tick
from matplotlib import pyplot as plt
from src.data.feff_data_raw import RAWDataFEFF
import random
from copy import deepcopy
import scienceplots
import matplotlib as mpl
import numpy as np
from src.data.feff_data import FEFFData
from src.data.vasp_data_raw import RAWDataVASP
from itertools import combinations_with_replacement
from pprint import pprint
from typing import Tuple, TypedDict, Union
import lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
import torch
import yaml
from scipy.signal import convolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scripts.pca_plots import linear_fit_of_pcas, plot_pcas
from scripts.plots_model_report import (
    heatmap_of_lines,
    plot_predictions,
    # plot_residue_histogram,
    plot_residue_cv,
    plot_residue_heatmap,
    plot_residue_quartiles,
)
from src.ckpt_predictions import get_optimal_fc_predictions
from src.plot.model_report import linear_model_predictions, model_report
from src.data.vasp_data import VASPData
from src.pl_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.plots.highlight_tick import highlight_tick

imports_1 = [
    "src",
    "utils.src.misc",
    "utils.src.plots.highlight_tick",
    "matplotlib.pyplot",
    "src.data.feff_data_raw",
    "scienceplots",
    "matplotlib",
    "src.data.feff_data",
    "src.data.vasp_data_raw",
    "scripts.pca_plots",
    "scripts.plots_model_report",
    "src.ckpt_predictions",
    "src.plot.model_report",
    "src.data.vasp_data",
    "src.pl_data",
    "utils.src.lightning.pl_module",
    "utils.src.optuna.dynamic_fc",
]


imports_2 = {
    "utils.src.misc": ["icecream"],
    "utils.src.plots.highlight_tick": ["highlight_tick"],
    "src.data.feff_data_raw": ["RAWDataFEFF"],
    "src.data.feff_data": ["FEFFData"],
    "src.data.vasp_data_raw": ["RAWDataVASP"],
    "src.data.vasp_data": ["VASPData"],
    "src.pl_data": ["XASData"],
    "utils.src.lightning.pl_module": ["PLModule"],
    "utils.src.optuna.dynamic_fc": ["PlDynamicFC"],
    "scripts.pca_plots": ["linear_fit_of_pcas", "plot_pcas"],
    "scripts.plots_model_report": [
        "heatmap_of_lines",
        "plot_predictions",
        "plot_residue_cv",
        "plot_residue_heatmap",
        "plot_residue_quartiles",
    ],
    "src.ckpt_predictions": ["get_optimal_fc_predictions"],
    "src.plot.model_report": ["linear_model_predictions", "model_report"],
}


def reimport_modules_and_functions():
    import importlib

    # Reimport non-standard modules
    for module in imports_1:
        importlib.reload(importlib.import_module(module))

    # Reimport specific functions/classes
    for module, items in imports_2.items():
        reloaded_module = importlib.import_module(module)
        globals().update({item: getattr(reloaded_module, item) for item in items})


reimport_modules_and_functions()

# %%

# id = ("mp-390", "000_Ti")
# vasp_raw_data = RAWDataVASP(compound=compound)
# processed_vasp_spectra = VASPDataModifier(vasp_raw_data.parameters[id])

# %%


compound = "Cu"
feff_raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)

sample_size = 5
seed = 42

feff_ids = set(feff_raw_data.parameters.keys())
vasp_ids = set(vasp_raw_data.parameters.keys())
common_ids = feff_ids.intersection(vasp_ids)

plt.style.use(["default", "science", "grid"])
random.seed(seed)
ids = random.choices(list(common_ids), k=sample_size)
fig, axs = plt.subplots(len(ids), 1, figsize=(8, 2 * len(ids)))
for simulation_type in ["VASP", "FEFF"]:
    raw_data = vasp_raw_data if simulation_type == "VASP" else feff_raw_data
    data_class = VASPData if simulation_type == "VASP" else FEFFData
    for ax, id in zip(axs, ids):
        data = data_class(
            compound=compound,
            params=raw_data.parameters[id],
        )

        if data.simulation_type == "FEFF":
            data.align(VASPData(compound, vasp_raw_data.parameters[id]))

        ax.plot(
            data.energy,
            data.spectra,
            label=f"{data.simulation_type} full",
            linestyle="--",
            color="red",
        )
        ax.legend()
        ax.sharex(axs[0])
axs[-1].set_xlabel("Energy (eV)", fontsize=18)

plt.suptitle(f"VASP truncation samples: {compound}", fontsize=18)
plt.tight_layout()
plt.savefig(f"vasp_truncation_examples_{compound}.pdf", bbox_inches="tight", dpi=300)
plt.show()

# ==============================================================================

# %%
