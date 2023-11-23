# %%

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

compound = "Cu"
feff_raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)

# %%

seed = 42
# seed = random.randint(0, 1000)
sample_size = 3
per_spectra_energy_alignment = False
feff_ids = set(feff_raw_data.parameters.keys())
vasp_ids = set(vasp_raw_data.parameters.keys())
common_ids = feff_ids.intersection(vasp_ids)
random.seed(seed)
ids = random.choices(list(common_ids), k=sample_size)


# %%

plt.style.use(["default", "science", "grid"])

fig, axs = plt.subplots(len(ids), 1, figsize=(8, 2 * len(ids)))
for simulation_type in ["VASP"]:
    raw_data = vasp_raw_data if simulation_type == "VASP" else feff_raw_data
    data_class = VASPData if simulation_type == "VASP" else FEFFData
    for ax, id in zip(axs, ids):
        data = data_class(
            compound=compound,
            params=raw_data.parameters[id],
        )
        if data.simulation_type == "FEFF" and per_spectra_energy_alignment:
            data.align(VASPData(vasp_raw_data.parameters[id]))

        ax.plot(
            data.energy,
            data.spectra,
            label=f"{data.simulation_type} full",
            linestyle="--",
        )
        data.truncate_emperically()
        ax.plot(
            data.energy,
            data.spectra,
            label=f"{data.simulation_type} chopped",
        )

        ax.legend()
        ax.sharex(axs[0])
plt.tight_layout()
# ==============================================================================

# %%

# %%

compound = "Cu"
simulation_type = "VASP"
raw_data = (
    RAWDataVASP(compound=compound) if simulation_type == "VASP" else feff_raw_data
)
data_class = VASPData if simulation_type == "VASP" else FEFFData

energy_start_list = []
count = 0
for id in raw_data.parameters.keys():
    print(count)
    count += 1
    if count > 200:
        break
    data = data_class(raw_data.parameters[id])
    energy_start = data.energy[0]
    energy_start_list.append(energy_start)
energy_start_list = np.array(energy_start_list)

# %%

fig = plt.figure(figsize=(8, 6))
plt.style.use(["default", "science"])
plt.rcParams["font.size"] = 16
plt.hist(
    energy_start_list, bins=20, facecolor="green", edgecolor="tab:green", alpha=0.5
)
plt.xlabel("Energy_start after truncation (eV)")
plt.ylabel("Count")
plt.title(
    f"Energy_start distribution for {compound} fot {simulation_type} \n Min: {min(energy_start_list):.2f} eV, Max: {max(energy_start_list):.2f} eV \n sample_size: {len(energy_start_list)} \n Median: {np.median(energy_start_list):.2f} eV"
)
plt.savefig(
    f"energy_start_distribution_{compound}_{simulation_type}.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%
