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


def extract_widths_from_ckpt_path(checkpoint_path):
    match = re.search(r"(\[.*\])", checkpoint_path)
    if match:
        extracted_str = match.group(1)
        widths = ast.literal_eval(extracted_str)
        return widths
    else:
        raise ValueError("No widths found in checkpoint path")


def load_model_from_ckpt(checkpoint_path, widths):
    # widths = extract_widths_from_ckpt_path(checkpoint_path)
    model = PlDynamicFC(widths=widths, output_size=200)
    checkpoint = torch.load(checkpoint_path)
    corrected_state_dict = {
        k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(corrected_state_dict)
    model.eval()
    return model


def fc_ckpt_predictions(ckpt_path, data_module, widths):
    model = load_model_from_ckpt(ckpt_path, widths)
    predictions = torch.tensor([])
    data = torch.tensor([])
    for batch in data_module.test_dataloader():
        x, y = batch
        data = torch.cat((data, y), dim=0)
        predictions = torch.cat((predictions, model(x)), dim=0)
    predictions = predictions.detach().numpy()
    data = data.detach().numpy()
    return f"fc_{model.widths}", data, predictions


if __name__ == "__main__":
    query = {
        "compound": "Cu-O",
        "simulation_type": "FEFF",
        "split": "material",
    }
    ckpt_path = "logs/[64]/runs/2023-10-25_17-55-19/lightning_logs/version_0/checkpoints/epoch=19-step=660.ckpt"
    data_module = XASData(query=query, batch_size=128, num_workers=0)
    predictions = fc_ckpt_predictions(ckpt_path, data_module)
