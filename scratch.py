# %%
from utils.src.plots.highlight_tick import highlight_tick
import multiprocessing
import re
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


def get_common_ids(vasp_raw_data, feff_raw_data):
    feff_ids = set(feff_raw_data.parameters.keys())
    vasp_ids = set(vasp_raw_data.parameters.keys())
    common_ids = feff_ids.intersection(vasp_ids)
    return common_ids


def random_sample(vasp_raw_data, feff_raw_data, seed):
    random.seed(seed)
    common_ids = get_common_ids(vasp_raw_data, feff_raw_data)
    id = random.choice(list(common_ids))
    vasp_data = VASPData(compound, vasp_raw_data.parameters[id])
    feff_data = FEFFData(compound, feff_raw_data.parameters[id])
    return vasp_data, feff_data


# %%

compound = "Ti"
feff_raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)

# %%

seed = 32
vasp_data, feff_data = random_sample(
    vasp_raw_data,
    feff_raw_data,
    seed=seed,
)

vasp_data.resample()
feff_data.resample()
kwargs = {"marker": "o", "markersize": 3}
plt.plot(vasp_data.energy, vasp_data.spectra, label="VASP", **kwargs)
plt.plot(feff_data.energy, feff_data.spectra, label="FEFF", **kwargs)


# %%

compound = "Ti"
feff_raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)

feff_ids = set(feff_raw_data.parameters.keys())
vasp_ids = set(vasp_raw_data.parameters.keys())
common_ids = feff_ids.intersection(vasp_ids)
