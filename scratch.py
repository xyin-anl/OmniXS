# %%
# %load_ext autoreload
# %autoreload 2
from src.model_report import model_report
import lightning as pl
import ast
import re
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
import numpy as np
import scienceplots
from scripts.plots_model_report import (
    plot_residue_quartiles,
    plot_residue_histogram,
    plot_residue_cv,
    plot_residue_heatmap,
    plot_predictions,
    heatmap_of_lines,
)
from utils.src.lightning.pl_module import PLModule
import torch
from pprint import pprint
from utils.src.optuna.dynamic_fc import PlDynamicFC
from src.ckpt_predictions import fc_ckpt_predictions

# %%


query = {
    "compound": "Cu-O",
    "simulation_type": "FEFF",
    "split": "material",
}
ckpt_path = "logs/Cu-O-feff/runs/2023-10-26_00-24-39/lightning_logs/version_0/checkpoints/epoch=93-step=3102.ckpt"
data_module = XASData(query=query, batch_size=1, num_workers=0)
model_report(
    query=query,
    model_fn=lambda query: fc_ckpt_predictions(
        ckpt_path, data_module, widths=[64, 180, 200]
    ),
)

# %%
query = {
    "compound": "Ti-O",
    "simulation_type": "FEFF",
    "split": "material",
}
ckpt_path = "logs/Ti-O-feff/runs/2023-10-26_00-24-39/lightning_logs/version_0/checkpoints/epoch=64-step=3445.ckpt"
data_module = XASData(query=query, batch_size=1, num_workers=0)
model_report(
    query=query,
    model_fn=lambda query: fc_ckpt_predictions(
        ckpt_path, data_module, widths=[64, 190, 180]
    ),
)

# %%
query = {
    "compound": "Ti-O",
    "simulation_type": "VASP",
    "split": "material",
}
ckpt_path = "logs/Ti-O-vasp/runs/2023-10-26_00-24-39/lightning_logs/version_0/checkpoints/epoch=30-step=744.ckpt"
data_module = XASData(query=query, batch_size=1, num_workers=0)
model_report(
    query=query,
    model_fn=lambda query: fc_ckpt_predictions(
        ckpt_path, data_module, widths=[64, 150, 120, 170]
    ),
)
# %%


train_data_X = []
for sim in ["FEFF", "VASP"]:
    query = {
        "compound": "Ti-O",
        "simulation_type": sim,
        "split": "material",
    }
    data = XASData.load_data(query=query)
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]
    train_data_X.append(X_train)
    # heatmap_of_lines(X_train)


# nohup python main.py compound_name=Cu-O simulation_type=feff model.model.widths=[64,180,200] trainer.max_epochs=500 model.learning_rate=0.000815876 >cu-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=feff model.model.widths=[64,190,180] trainer.max_epochs=500 model.learning_rate=0.001069577 >ti-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=vasp model.model.widths=[64,150,120,170] trainer.max_epochs=500 model.learning_rate=0.00436152122 >ti-o-vasp.log &

# %%
# summary of train_data_X
x1 = pd.DataFrame(train_data_X[0])
x2 = pd.DataFrame(train_data_X[1])

(x1.describe(percentiles=None))

# %%

(x2.describe(percentiles=None))
# %%
