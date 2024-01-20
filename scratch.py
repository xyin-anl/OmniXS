# %%
import matplotlib.lines as mlines
import optuna
from typing import Dict
from src.data.ml_data import DataQuery
from utils.src.misc.icecream import ic
from src.data.ml_data import XASPlData
from typing import Tuple, TypedDict, Union
import warnings
from src.data.material_split import MaterialSplitter
import os
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from torch.utils.data import TensorDataset
from utils.src.lightning.pl_data_module import PlDataModule

from config.defaults import cfg
import re
import pickle
from p_tqdm import p_map
from src.data.raw_data import RAWData
from scripts.data_scripts.m3gnet_version_fix import fix_m3gnet_version
from utils.src.plots.highlight_tick import highlight_tick
import multiprocessing
from tqdm import tqdm
import time
import pickle
import importlib
import src
from utils.src.misc import icecream
from matplotlib import pyplot as plt
from src.data.feff_data_raw import RAWDataFEFF
import random
from copy import deepcopy
import scienceplots
import matplotlib as mpl
from src.data.feff_data import FEFFData
from src.data.vasp_data_raw import RAWDataVASP
from itertools import combinations_with_replacement
from pprint import pprint
import lightning as pl
import pandas as pd
import seaborn as sns
import yaml
from scipy.signal import convolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scripts.model_scripts.pca_plots import linear_fit_of_pcas, plot_pcas
from scripts.model_scripts.plots_model_report import (
    heatmap_of_lines,
    plot_predictions,
    # plot_residue_histogram,
    plot_residue_cv,
    plot_residue_heatmap,
    plot_residue_quartiles,
)
from src.models.ckpt_predictions import get_optimal_fc_predictions
from scripts.model_scripts.model_report import linear_model_predictions, model_report
from src.data.vasp_data import VASPData

# from src.pl_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from src.data.ml_data import DataQuery, XASPlData

# %%

from functools import cached_property
from abc import ABC, abstractmethod
from src.data.ml_data import XASPlData
from sklearn.linear_model import LinearRegression
import optuna
import numpy as np
from config.defaults import cfg
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.data.ml_data import XASPlData, load_xas_ml_data


class TrainedModel(ABC):
    def __init__(self, compound, simulation_type="FEFF"):
        self.compound = compound
        self.simulation_type = simulation_type

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predictions(self):
        pass

    @cached_property
    def mae_per_spectra(self):
        return np.mean(np.abs(self.data.test.y - self.predictions), axis=1)

    @cached_property
    def mae(self):
        return mean_absolute_error(self.data.test.y, self.predictions)

    def sorted_predictions(self, sort_array=None):
        sort_array = sort_array or self.mae_per_spectra  # default sort by mae
        pair = np.column_stack((self.data.test.y, self.predictions))
        pair = pair.reshape(-1, 2, self.data.test.y.shape[1])
        pair = pair[sort_array.argsort()]
        return pair

    def top_predictions(self, splits=10):
        # sort by mean residue, splits and return top of each split
        pair = self.sorted_predictions()
        # for even split, some pairs are chopped off
        new_len = len(pair) - divmod(len(pair), splits)[1]
        pair = pair[:new_len]
        top_spectra = [s[0] for s in np.split(pair, splits)]
        return np.array(top_spectra)

    @cached_property
    def absolute_errors(self):
        return np.abs(self.data.test.y - self.predictions)

    @cached_property
    def data(self):
        return load_xas_ml_data(
            query=DataQuery(
                compound=self.compound,
                simulation_type=self.simulation_type,
            )
        )

    @cached_property
    def peak_errors(self):
        max_idx = np.argmax(model.data.test.y, axis=1)
        peak_errors = np.array(
            [error[idx] for error, idx in zip(model.absolute_errors, max_idx)]
        )
        return peak_errors


class LinReg(TrainedModel):
    name = "LinReg"

    @cached_property
    def model(self):
        return LinearRegression().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class Trained_FCModel(TrainedModel):
    name = "FCModel"

    @cached_property
    def model(self):
        # load from ckpt or build from params
        raise NotImplementedError

    @cached_property
    def optuna_study(self):
        kwargs = dict(
            compound=self.compound,
            simulation_type=self.simulation_type,
        )
        study = optuna.load_study(
            study_name=cfg.optuna.study_name.format(**kwargs),
            storage=cfg.paths.optuna_db.format(**kwargs),
        )
        return study

    @cached_property
    def losses(self):
        return self.optuna_study.best_value

    @cached_property
    def residues(self):
        raise NotImplementedError


class Plot:
    def __init__(self):
        self.style()
        self.title = None

    @cached_property
    def colors_for_compounds(self):
        compounds = cfg.compounds
        colormap = plt.cm.get_cmap("tab10")
        color_idx = np.linspace(0, 1, len(compounds))
        compound_colors = {c: colormap(color_idx[i]) for i, c in enumerate(compounds)}
        return compound_colors

    def style(self):
        plt.style.use(["default", "science"])
        return self

    def reset(self):
        plt.style.use("default")
        self.title = None
        return self

    def set_title(self, title):
        self.title = title
        return self

    def plot_top_predictions(self, top_predictions, splits=5, axs=None):
        if axs is None:
            fig, axs = plt.subplots(splits, 1, figsize=(8, 20))
            axs = axs.flatten()
        else:
            assert len(axs) == splits, "Number of subplots must match splits"

        for i, ax in enumerate(axs):
            t = top_predictions[i][0]
            p = top_predictions[i][1]

            ax.plot(t, "-", color="black", linewidth=1.5)
            ax.plot(p, "--", color="red", linewidth=1.5)

            # ax.fill_between(
            #     np.arange(len(top_predictions[i][0])),
            #     t,
            #     p,
            #     alpha=0.4,
            #     label=f"Mean Residue {(t-p).__abs__().mean():.1e}",
            #     color="red",
            # )

            ax.set_axis_off()

        # remove axis and ticks other than x-axis in last plot
        axs[-1].set_axis_on()
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        axs[-1].spines["left"].set_visible(False)
        axs[-1].tick_params(axis="y", which="both", left=False, labelleft=False)

        # set x-axis ticks labels based on cfg values
        e_start = cfg.transformations.e_start[compound]
        e_diff = cfg.transformations.e_range_diff
        e_end = e_start + e_diff
        axs[-1].set_xticks(np.linspace(0, len(top_predictions[0][0]), 2))
        axs[-1].set_xticklabels(
            [
                f"{e:.1f}"
                for e in np.linspace(e_start, e_end, len(axs[-1].get_xticklabels()))
            ],
            fontsize=14,
        )
        axs[-1].set_xlabel(self.title, fontsize=20)

        # self.title = f"Top {splits} predictions of {top_predictions}"
        # if self.title is not None:
        #     plt.suptitle(self.title, fontsize=20, y=1.005)
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.tight_layout()
        return self

    def save(self):
        file_name = self.title.replace(" ", "_").lower() if self.title else "untitled"
        plt.savefig(f"{file_name}.pdf", bbox_inches="tight", dpi=300)
        return self


# %%


splits = 10
top_predictions = {c: LinReg(c).top_predictions(splits=splits) for c in cfg.compounds}

# %%
# compound = cfg.compounds[0]
# model = LinReg(compound)

fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(20, 16))
for i, compound in enumerate(cfg.compounds):
    Plot().set_title(f"{compound}").plot_top_predictions(
        top_predictions[compound], splits=splits, axs=axs[:, i]
    )

# plt.subplots_adjust(wspace=0, hspace=0.05)
# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout(pad=0)
plt.savefig(f"top_{splits}_predictions.pdf", bbox_inches="tight", dpi=300)

# %%
