# %%
# %load_ext autoreload
# %autoreload 2

# from scripts.plots_model_report import plot_residue_histogram
from scipy.stats import cauchy
import numpy as np
from scipy.signal import convolve

import numpy as np
from utils.src.plots.highlight_tick import highlight_tick
import yaml
import scienceplots
from src.model_report import linear_model_predictions
from src.model_report import model_report
import lightning as pl
import ast
from scripts.pca_plots import plot_pcas, linear_fit_of_pcas
import pandas as pd
import seaborn as sns
from src.xas_data import XASData
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scienceplots
from scripts.plots_model_report import (
    plot_residue_quartiles,
    # plot_residue_histogram,
    plot_residue_cv,
    plot_residue_heatmap,
    plot_predictions,
    heatmap_of_lines,
)
from utils.src.lightning.pl_module import PLModule
import torch
from pprint import pprint
from utils.src.optuna.dynamic_fc import PlDynamicFC
from src.ckpt_predictions import get_optimal_fc_predictions
from typing import TypedDict, Union, Tuple
from src.xas_data_raw import RAWData


# %%

## MISSING DATA
# for compound in ["Ti", "Cu"]:
#     data = RAWData(compound, "VASP")
#     print(
#         f"RAW VASP data for {compound}: \
#             {len(data)}, {len(data.missing_data)} (missing)"
#     )
#     with open(f"missing_VASP_data_{compound}.txt", "w") as f:
#         for d in data.missing_data:
#             f.write(f"{d}\n")

# %%

# compound = "Ti"
compound = "Cu"
data = RAWData(compound, "VASP")

# %%
id = next(iter(data.parameters))
single_spectra = data.parameters[id]["mu"]
energy = single_spectra[:, 0]
spectra = single_spectra[:, 1]
e_core = data.parameters[id]["e_core"]
e_cbm = data.parameters[id]["e_cbm"]
E_ch = data.parameters[id]["E_ch"]
E_GS = data.parameters[id]["E_GS"]
volume = data.parameters[id]["volume"]

print("-" * 80)
print(f"compound: {compound}")
print("simulation_type: VASP")
print(f"e_core: {e_core}")
print(f"e_cbm: {e_cbm}")
print(f"E_ch: {E_ch}")
print(f"E_GS: {E_GS}")
print(f"volume: {volume}")
print("-" * 80)


# %%

print(f"Lenght of spectra: {len(spectra)}")
print(f"Lenght of Non-zero spectra: {len(spectra[spectra != 0])}")
print(f"Lenght of Zero spectra: {len(spectra[spectra == 0])}")

# %%

plt.plot(spectra[spectra != 0], label="Non-zero")
plt.plot(spectra, label="All")
plt.legend()

# TODO: double check this

# %%

first_non_zero_idx = np.where(spectra != 0)[0][0]
last_non_zero_idx = np.where(spectra != 0)[0][-1]
offset = 200
non_zero_idx = np.arange(
    first_non_zero_idx + offset,
    last_non_zero_idx - offset,
    dtype=int,
)
assert np.all(np.diff(non_zero_idx) == 1), "non_zero_idx is not continuous"
spectra = spectra[non_zero_idx]
energy = energy[non_zero_idx]

# check if there are zeros in between

# %%

## ALIGNMENT
# offset = (e_core - e_cbm) + (E_ch - E_GS)
offset = (0 - e_cbm) + (E_ch - E_GS)
# TODO: e_core might be wrong
print(f"Energy Offset: {offset}")
energy_aligned = energy + offset
plt.plot(energy_aligned, spectra, label="Aligned")
plt.legend()

# %%

## SCALING
omega = spectra * energy_aligned
big_omega = volume
alpha = 1 / 137  # TODO: ask if it is fine-structure constant or something else
scaled_amplitude = (omega * big_omega) / alpha  # ASK if alpha is mul or div
plt.plot(energy_aligned, scaled_amplitude, label="Scaled and Aligned")
plt.legend()


# %%


# %%


## BRODENING
# Source: https://github.com/AI-multimodal/Lightshow/blob/mc-broadening-revisions/lightshow/postprocess/broaden.py
def lorentz_broaden(x, xin, yin, gamma):
    """Lorentzian broadening function

    Parameters
    ----------
    x : numpy.ndarray
        The output energy grid.
    xin : numpy.ndarray
        The input energy grid.
    yin : numpy.ndarray
        The input spectrum.
    gamma : float
        Lorentzian broadening gamma.

    Returns
    -------
    numpy.ndarray
        Broadened spectrum.
    """

    x1, x2 = np.meshgrid(x, xin)
    dx = xin[-1] - xin[0]
    return np.dot(cauchy.pdf(x1, x2, gamma).T, yin) / len(xin) * dx


broadened_amplitude = lorentz_broaden(
    energy_aligned,
    energy_aligned,
    scaled_amplitude,
    gamma=0.89 / 2,  # TODO: mention this
)

# %%

plt.plot(energy_aligned, scaled_amplitude, label="Scaled and Aligned")
plt.plot(energy_aligned, broadened_amplitude, label="Scaled, Aligned and Broadened 2")
plt.legend()
# %%

fig, ax = plt.subplots(4, 1)
ax[0].plot(energy, spectra, label="Raw +- 200 (non-zero)")
ax[1].plot(energy_aligned, scaled_amplitude, label="Scaled and Aligned")
ax[2].plot(energy_aligned, broadened_amplitude, label="Scaled, Aligned and Broadened")
ax[1].sharex(ax[2])

# %%
