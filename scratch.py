# %%
# %load_ext autoreload
# %autoreload 2

# from scripts.plots_model_report import plot_residue_histogram
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
data = RAWData("Cu", "VASP")

# %%
print("-" * 50)
print(data._ids[:5])
print(len(data._ids))
print("-" * 50)

# %%
print("-" * 50)
print(data._sites[data._ids[0]])
print(len(data._sites[data._ids[0]]))
print("-" * 50)

# %%
print("-" * 50)
print(data._mu[data._ids[0], data._sites[data._ids[0]][0]])
print(len(data._mu[data._ids[0], data._sites[data._ids[0]][0]]))
print("-" * 50)

# %%
print("-" * 50)
print(data._e_cbm[data._ids[0], data._sites[data._ids[0]][0]])
print("-" * 50)


# %%

print("-" * 50)
print(data._E_ch)
print("-" * 50)

# %%

print(
    data.missing_data
    # data.missing_e_cbm,
    # data.missing_E_ch,
    # data.missing_E_GS,
)


# %%

data._ids

# %%
data.missing_data

# %%
missing_ids = [id for id, site in data.missing_data]

# %%
set([1,2])


# %%
len(set(missing_ids))
# %%
