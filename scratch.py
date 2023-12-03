# %%
from tqdm import tqdm
import os
import time
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

    for module in imports_1:
        importlib.reload(importlib.import_module(module))
    for module, items in imports_2.items():
        reloaded_module = importlib.import_module(module)
        globals().update({item: getattr(reloaded_module, item) for item in items})


reimport_modules_and_functions()

# %%

# id = ("mp-390", "000_Ti")
# vasp_raw_data = RAWDataVASP(compound=compound)
# processed_vasp_spectra = VASPDataModifier(vasp_raw_data.parameters[id])

# %%

compound = "Ti"
raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)


# %%

# seed = 42
# random.seed(seed)
sample_size = 10
ids = set(raw_data.parameters.keys())
vasp_ids = set(vasp_raw_data.parameters.keys())
common_ids = ids.intersection(vasp_ids)

plt.style.use(["default", "science"])
ids = random.choices(list(common_ids), k=sample_size)
fig, axs = plt.subplots(len(ids), 1, figsize=(6, 3 * len(ids)))
time_corr = []
time_dtw = []
for simulation_type in ["VASP", "FEFF"]:
    raw_data = vasp_raw_data if simulation_type == "VASP" else raw_data
    data_class = VASPData if simulation_type == "VASP" else FEFFData
    for ax, id in zip(axs, ids):
        data = data_class(
            compound=compound,
            params=raw_data.parameters[id],
        )
        # data.transform()

        if data.simulation_type == "FEFF":
            t1 = time.time()
            data.align(VASPData(compound, vasp_raw_data.parameters[id]))
            del_t = time.time() - t1
            time_corr.append(del_t)
            ic(del_t)
        data.truncate_emperically()
        ax.plot(
            data.energy,
            data.spectra,
            label=f"{data.simulation_type}_{id}",
            linestyle="-",
        )

        # doing again for dtw based
        if data.simulation_type == "FEFF":
            data = data_class(
                compound=compound,
                params=raw_data.parameters[id],
            )
            t1 = time.time()
            shift = data_class.dtw_shift(
                data,
                VASPData(compound, vasp_raw_data.parameters[id]),
            )
            del_t = time.time() - t1
            time_dtw.append(del_t)
            ic(del_t)
            data.align_energy(-shift)
            data.truncate_emperically()
            ax.plot(
                data.energy,
                data.spectra,
                label=f"{data.simulation_type}_{id}_dtw",
                linestyle="--",
            )

        ax.legend()
        ax.sharex(axs[0])

axs[-1].set_xlabel("Energy (eV)", fontsize=18)

plt.suptitle(f"Per-spectra alignment samples: {compound}", fontsize=18)
plt.tight_layout()
# plt.savefig(f"vasp_truncation_examples_{compound}.pdf", bbox_inches="tight", dpi=300)
plt.show()

# ==============================================================================

# %%

# compound = "Ti"
# simulation_type = "FEFF"
# raw_data_class = RAWDataFEFF if simulation_type == "FEFF" else RAWDataVASP
# raw_data = raw_data_class(compound=compound)
# ids = raw_data.parameters.keys()
# ids = tqdm(ids)  # progress bar
# for id in ids:
#     data_class = FEFFData if simulation_type == "FEFF" else VASPData
#     data = data_class(
#         compound=compound,
#         params=raw_data.parameters[id],
#         id=id,
#     )
#     data.save()
#     # plt.plot(data.energy, data.spectra, label=id)
#     # plt.legend()
# # plt.show()

# %%

# %%

data_vasp = VASPData(compound).load(id=("mp-390", "000_Ti"))
data_feff = FEFFData(compound).load(id=("mp-390", "000_Ti"))

plt.plot(data_vasp.energy, data_vasp.spectra, label="VASP")
plt.plot(data_feff.energy, data_feff.spectra, label="FEFF")
plt.legend()
plt.show()


# %%
