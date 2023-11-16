# %%
# %load_ext autoreload
# %autoreload 2
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
from src.plot_vasp_transormations import VASPDataTransformationPlotter
from src.raw_data_feff import RAWDataFEFF
from src.vasp_data_transformations import VASPDataModifier
from src.xas_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.plots.highlight_tick import highlight_tick


compound = "Ti"
id = ("mp-390", "000_Ti")
vasp = RAWDataVASP(compound=compound)
vasp_spectra = VASPDataModifier(vasp.parameters[id])

# feff = RAWDataFEFF(compound=compound)
# feff_spectra = FEFFDataModifier(feff.parameters[id])
# feff_spectra.align_to_spectra(vasp_spectra)


energy_values = []
count = 0
for spectra_data in vasp.parameters.values():
    count += 1
    if count > 100:
        break
    modifier = VASPDataModifier(spectra_data)
    start_energy = modifier.energy[0]
    end_energy = modifier.energy[-1]
    energy_values.append((start_energy, end_energy))

    # Print the current state
    print(
        f"Processed {len(energy_values)} out of {len(vasp.parameters)}: Start energy = {start_energy}, End energy = {end_energy}"
    )

energy_start_list, energy_end_list = zip(*energy_values)
energy_start_list = np.array(energy_start_list)
energy_end_list = np.array(energy_end_list)

energy_start_list, energy_end_list = zip(*energy_values)
energy_start_list = np.array(energy_start_list)
energy_end_list = np.array(energy_end_list)

np.save("energy_start_list.npy", energy_start_list)
np.save("energy_end_list.npy", energy_end_list)


plt.hist(energy_start_list, bins=100)
plt.hist(energy_end_list, bins=100)
print(f"maximum start energy: {np.max(energy_start_list)}")
print(f"minimum end energy: {np.min(energy_end_list)}")

# %%
