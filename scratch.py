# %%
# %load_ext autoreload
# %autoreload 2
import random
from copy import deepcopy
import scienceplots
import matplotlib as mpl

import numpy as np
from src.feff_data_transformations import FEFFDataModifier

# from scripts.plots_model_report import plot_residue_histogram
from src.raw_data_vasp import RAWDataVASP
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
from src.model_report import linear_model_predictions, model_report
from scripts.plot_vasp_transormations import VASPDataTransformationPlotter
from src.raw_data_feff import RAWDataFEFF
from src.vasp_data_transformations import VASPDataModifier
from src.xas_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.plots.highlight_tick import highlight_tick


# %%


compound = "Ti"
id = ("mp-390", "000_Ti")
vasp_raw_data = RAWDataVASP(compound=compound)
processed_vasp_spectra = VASPDataModifier(vasp_raw_data.parameters[id])

# %%

# ==============================================================================
# RANDOM SMAPLES OF FINAL PROCESSED VASP DATA WITH FILTERED ENERGY RANGE
# ==============================================================================
sample_size = 5
ids = random.choices(list(vasp_raw_data.parameters.keys()), k=sample_size)
# 4960 + 40(or 45) eV:
energy_range = [4960, 4960 + 50]
plt.style.use(["default", "science", "grid"])
fig, axs = plt.subplots(len(ids), 1, figsize=(8, 2 * len(ids)))
for ax, id in zip(axs, ids):
    processed_vasp_spectra = VASPDataModifier(vasp_raw_data.parameters[id])
    full_spectra = deepcopy(processed_vasp_spectra)
    ax.plot(full_spectra.energy, full_spectra.spectra, label="full", linestyle="--")
    chopped_spectra = deepcopy(processed_vasp_spectra).filter(energy_range=energy_range)
    ax.plot(chopped_spectra.energy, chopped_spectra.spectra, label="chopped")
    ax.legend()
    ax.sharex(axs[0])
plt.suptitle(f"Random sample for VASP spectra for {compound}")
plt.tight_layout()
plt.savefig(f"vasp_range_filter_examples_{compound}.pdf", bbox_inches="tight", dpi=300)
# ==============================================================================

# %%

# %%
