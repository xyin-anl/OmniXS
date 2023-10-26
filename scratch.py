# %%
# %load_ext autoreload
# %autoreload 2
import scienceplots
from src.model_report import linear_model_predictions
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
# %%
X1, x2 = train_data_X

# %%
pd.DataFrame(x2).describe()

# %%

x1 = X1[: len(x2)]

# %%

heatmap_of_lines(x2)

# %%

# heatmap_of_lines(x1)
# heatmap_of_lines(x1)
# heatmap_of_lines(x2)

heatmap_of_lines(x1 - x2, ylabel="tio_feff_train[:len(..)] - tio_vasp_train")


# %%


def plot_best_worst_residues(query, count=10):
    model_name, y_test, predictions = linear_model_predictions(query)
    data = XASData.load_data(query=query)
    _, _ = data["train"]["X"], data["train"]["y"]
    _, y_test = data["test"]["X"], data["test"]["y"]

    y_residues = np.abs(y_test - predictions)
    df_residues = pd.DataFrame(y_residues.mean(axis=1), columns=["residues"])
    df_test_data = pd.DataFrame(y_test)
    df_predictions = pd.DataFrame(predictions)
    df_all = pd.concat([df_residues, df_test_data, df_predictions], axis=1)

    column_names = (
        ["residues"]
        + [f"test_data_{i}" for i in range(y_test.shape[1])]
        + [f"predictions_{i}" for i in range(y_test.shape[1])]
    )

    df_all.columns = column_names
    df_all = df_all.sort_values(by="residues")

    df_data = df_all[[col for col in column_names if "data" in col]]
    df_predictions = df_all[[col for col in column_names if "predictions" in col]]

    plt.style.use(["science", "notebook", "high-vis", "no-latex"])
    fig, axes = plt.subplots(count, 2, figsize=(20, 20), sharex=True, sharey=True)

    for i in range(count):
        # Plot best on the left column
        df_data.iloc[i].plot(ax=axes[i, 0], label="data")
        df_predictions.iloc[i].plot(ax=axes[i, 0], label="predictions", linestyle="--")
        axes[i, 0].set_title(f"MAE: {round(df_all.iloc[i]['residues'], 3)}")

        # Plot worst on the right column
        idx = len(df_all) - 1 - i
        df_data.iloc[idx].plot(ax=axes[i, 1], label="data")
        df_predictions.iloc[idx].plot(
            ax=axes[i, 1], label="predictions", linestyle="--"
        )
        axes[i, 1].set_title(f"MAE: {round(df_all.iloc[idx]['residues'], 3)}")

        # Remove ticks and labels
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_yticklabels([])
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_yticklabels([])

    # Add common labels and titles
    plt.suptitle(
        f"{query['compound']}_{query['simulation_type']}",
        fontsize=40,
    )
    fig.text(0.25, 0.98, "Best Residues", ha="center", fontsize=30, c="green")
    fig.text(0.75, 0.98, "Worst Residues", ha="center", fontsize=30, c="red")
    fig.text(0.5, 0.04, "Common X-axis", ha="center")
    fig.text(0.04, 0.5, "Common Y-axis", va="center", rotation="vertical")

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    plt.legend()
    plt.savefig(f"{query['compound']}_{query['simulation_type']}_top_residues.pdf")


for compound in ["Cu-O", "Ti-O"]:
    for simulation_type in ["FEFF", "VASP"]:
        if simulation_type == "VASP" and compound == "Cu-O":
            continue
        plot_best_worst_residues(
            query={
                "compound": compound,
                "simulation_type": simulation_type,
                "split": "material",
            },
            count=10,
        )

# %%
