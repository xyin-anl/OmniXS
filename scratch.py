# %%
# %load_ext autoreload
# %autoreload 2
from utils.src.plots.highlight_tick import highlight_tick
from matplotlib import pyplot as plt
from src.data.feff_data_raw import RAWDataFEFF

import random
from copy import deepcopy
import scienceplots
import matplotlib as mpl

import numpy as np
from src.data.feff_data import FEFFData

# from scripts.plots_model_report import plot_residue_histogram
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
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data import VASPData
from src.pl_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.plots.highlight_tick import highlight_tick


# %%


# compound = "Ti"
# id = ("mp-390", "000_Ti")
# vasp_raw_data = RAWDataVASP(compound=compound)
# processed_vasp_spectra = VASPDataModifier(vasp_raw_data.parameters[id])

# %%

# # ==============================================================================
# # RANDOM SMAPLES OF FINAL PROCESSED VASP DATA WITH FILTERED ENERGY RANGE
# # ==============================================================================
# sample_size = 5
# ids = random.choices(list(vasp_raw_data.parameters.keys()), k=sample_size)
# # 4960 + 40(or 45) eV:
# energy_range = [4960, 4960 + 50]
# plt.style.use(["default", "science", "grid"])
# fig, axs = plt.subplots(len(ids), 1, figsize=(8, 2 * len(ids)))
# for ax, id in zip(axs, ids):
#     processed_vasp_spectra = VASPDataModifier(vasp_raw_data.parameters[id])
#     full_spectra = deepcopy(processed_vasp_spectra)
#     ax.plot(full_spectra.energy, full_spectra.spectra, label="full", linestyle="--")
#     chopped_spectra = deepcopy(processed_vasp_spectra).filter(energy_range=energy_range)
#     ax.plot(chopped_spectra.energy, chopped_spectra.spectra, label="chopped")
#     ax.legend()
#     ax.sharex(axs[0])
# plt.suptitle(f"Random sample for VASP spectra for {compound}")
# plt.tight_layout()
# plt.savefig(f"vasp_range_filter_examples_{compound}.pdf", bbox_inches="tight", dpi=300)
# # ==============================================================================

# %%

# %%

compound = "Ti"
feff_raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)


# %%

seed = 42
sample_size = 5
per_spectra_energy_alignment = True
common_ids = set(feff_raw_data.parameters.keys()).intersection(
    set(vasp_raw_data.parameters.keys())
)
random.seed(seed)
ids = random.choices(list(common_ids), k=sample_size)
energy_range = [4960, 4960 + 40]  # Ti
energy_range = [4600, 4600 + 40]  # Cu
plt.style.use(["default", "science", "grid"])

sup_title = f"Random sample of processed spectras for {compound}"

if per_spectra_energy_alignment:
    sup_title += "\neach spectra shifted by different amount to align with VASP"
else:
    sup_title += "\nall spectra shifted by same amount to align with VASP"

fig, axs = plt.subplots(len(ids), 1, figsize=(8, 2 * len(ids)))
for simulation_type in ["FEFF", "VASP"]:
    raw_data = vasp_raw_data if simulation_type == "VASP" else feff_raw_data
    data_modifier = VASPDataModifier if simulation_type == "VASP" else FEFFDataModifier
    for ax, id in zip(axs, ids):
        processed_spectra = data_modifier(raw_data.parameters[id])
        if compound == "Cu" and simulation_type == "VASP":  # TODO: change this
            processed_spectra.reset().truncate().scale().broaden(gamma=1.19).align()
        if simulation_type == "FEFF" and per_spectra_energy_alignment:
            processed_spectra._energy = (
                processed_spectra.energy
                - processed_spectra.spearman_align(
                    VASPDataModifier(vasp_raw_data.parameters[id])
                )
            )
        full_spectra = deepcopy(processed_spectra)
        chopped_spectra = deepcopy(processed_spectra).filter(energy_range=energy_range)
        ax.plot(
            full_spectra.energy,
            full_spectra.spectra,
            label=f"{simulation_type} full",
            linestyle="--",
        )
        ax.plot(
            chopped_spectra.energy,
            chopped_spectra.spectra,
            label=f"{simulation_type} chopped",
        )
        ax.legend()
        ax.sharex(axs[0])
plt.suptitle(sup_title)
plt.tight_layout()
plt.savefig(
    f"range_filter_examples_{compound}_{per_spectra_energy_alignment}.pdf",
    bbox_inches="tight",
    dpi=300,
)
# ==============================================================================

# %%

# temp = pearson_alignments
# plt.hist(temp)
# %%

np.load("pearson_shifts.npy").mean()
