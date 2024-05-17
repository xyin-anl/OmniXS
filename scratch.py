# %%
from scripts.model_scripts.plot_universal_tl_model import (
    plot_universal_tl_vs_per_compound_tl,
)
from src.models.trained_models import MeanModel
from scipy.stats import gaussian_kde
from config.defaults import cfg
import scienceplots
from matplotlib import pyplot as plt
import random
from plot_top_pred_comparisions import plot_top_pred_comparisions
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from scripts.plots.plot_all_spectras import MLDATAPlotter
from src.data.ml_data import load_all_data
from src.models.trained_models import Trained_FCModel
from src.data.ml_data import DataQuery, load_xas_ml_data
import numpy as np


# %%
