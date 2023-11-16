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


# %%

compound = "Ti"
id = ("mp-390", "000_Ti")
vasp = RAWDataVASP(compound=compound)
vasp_spectra = VASPDataModifier(vasp.parameters[id])

feff = RAWDataFEFF(compound=compound)
feff_spectra = FEFFDataModifier(feff.parameters[id])
feff_spectra.align_to_spectra(vasp_spectra)


# %%


xs_mp_390 = np.load("dataset/misc/xs-mp-390.npy")
xs_energy, xs_spectra = xs_mp_390[0], xs_mp_390[1]
xs_energy -= (
    xs_energy[np.argmax(xs_spectra)]
    - vasp_spectra.energy[np.argmax(vasp_spectra.spectra)]
)  # align to vasp max

# mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use(["vibrant", "no-latex"])
plt.figure(figsize=(8, 5))
plt.plot(vasp_spectra.energy, vasp_spectra.spectra, label="vasp", c="green")
plt.plot(feff_spectra.energy, feff_spectra.spectra, label="feff")
plt.plot(xs_energy, xs_spectra, label="xs", color="orange")
# plt.xlim(vasp_spectra.energy[0], vasp_spectra.energy[-1])
plt.legend()

# %%
