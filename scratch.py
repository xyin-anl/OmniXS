# %%
# %load_ext autoreload
# %autoreload 2

# from scripts.plots_model_report import plot_residue_histogram
from scipy.signal import convolve
from src.raw_data_feff import RAWDataFEFF

from src.vasp_data_transformations import VASPDataModifier
from src.plot_vasp_transormations import VASPDataTransformationPlotter
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



# %%


data = RAWDataFEFF(compound="Ti")
data.parameters
print(f"data: {len(data)}, missing: {len(data.missing_data)}")

id = next(iter(data._ids))
site = next(iter(data._sites[id]))
id = ("mp-390", "000_Ti")  # reference to another paper data

plt.plot(
    data.parameters[id]["mu"][0],
    data.parameters[id]["mu"][1],
)
plt.show()

# %%
