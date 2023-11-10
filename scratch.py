# %%
# %load_ext autoreload
# %autoreload 2
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


# %%

feff = RAWDataFEFF(compound="Ti")
feff_spectra = FEFFDataModifier(feff.parameters[("mp-390", "000_Ti")])

# %%
vasp = RAWDataVASP(compound="Ti")
vasp_spectra = VASPDataModifier(vasp.parameters[("mp-390", "000_Ti")])
# %%

# v = vasp.parameters[("mp-390", "000_Ti")]["mu"]
# f = feff.parameters[("mp-390", "000_Ti")]["mu"]

# %%

e_v = vasp_spectra.energy_trunc
e_f = feff_spectra.energy
s_v = vasp_spectra.spectra
s_f = feff_spectra.spectra
print(f"vasp_eneryy: {len(e_v)}, feff_energy: {len(e_f)}")
print(f"vasp_range: {e_v[0]} - {e_v[-1]}")
print(f"feff_range: {e_f[0]} - {e_f[-1]}")
print(f"vasp diff: {e_v[-1] - e_v[0]}")
print(f"feff diff: {e_f[-1] - e_f[0]}")
lower_diff = np.min((e_v[-1] - e_v[0], e_f[-1] - e_f[0]))
print(f"lower diff: {lower_diff}")
e_v = e_v - e_v[0]
e_f = e_f - e_f[0]
print(f"vasp_range: {e_v[0]} - {e_v[-1]}")
print(f"feff_range: {e_f[0]} - {e_f[-1]}")
vasp_range = e_v[-1] - e_v[0]
feff_range = e_f[-1] - e_f[0]
print(f"vasp diff: {vasp_range}")
print(f"feff diff: {feff_range}")
# union of two energy ranges

#%%

mu_v = np.array([e_v, s_v]).T
mu_f = np.array([e_f, s_f]).T
print(f"mu_v: {mu_v.shape}, mu_f: {mu_f.shape}")

# %%
