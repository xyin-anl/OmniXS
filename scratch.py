# %%
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
from src.data.ml_data import DataQuery

# %%

from functools import cached_property
from abc import ABC, abstractmethod
from src.data.ml_data import XASPlData
from sklearn.linear_model import LinearRegression
import optuna
import numpy as np
from config.defaults import cfg


class TrainedModel(ABC):
    def __init__(self, compound, simulation_type):
        self.compound = compound
        self.simulation_type = simulation_type

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def residues(self):
        pass

    @staticmethod
    def get_data(compound, simulation_type):
        data = XASData(
            query=DataQuery(
                compound=compound,
                simulation_type=simulation_type,
            ),
        )
        X_train, y_train = data.train_dataset.tensors
        X_test, y_test = data.test_dataset.tensors
        return X_train, y_train, X_test, y_test


class LinReg(TrainedModel):
    @property
    def name(self):
        return "LinReg"

    @cached_property
    def model(self):
        X_train, y_train, _, _ = self.get_data(
            compound=self.compound, simulation_type=self.simulation_type
        )
        fitted_model = LinearRegression().fit(X_train, y_train)
        return fitted_model

    @cached_property
    def residues(self):
        _, _, X_test, y_test = self.get_data(
            compound=self.compound,
            simulation_type=self.simulation_type,
        )
        y_pred = self.model.predict(X_test)
        y_residues = (y_test - y_pred).flatten()
        return y_residues


class Trained_FCModel(TrainedModel):
    @property
    def name(self):
        return "FCModel"

    @cached_property
    def model(self):
        # load from ckpt or build from params
        raise NotImplementedError

    @cached_property
    def _optuna_study(self):
        optuna_study_storage = cfg.paths.optuna_db.format(
            compound=self.compound,
            simulation_type=self.simulation_type,
        )
        study = optuna.load_study(
            study_name=f"{self.compound}-{self.simulation_type}",
            storage=optuna_study_storage,
        )
        return study

    @cached_property
    def losses(self):
        return self._optuna_study.best_value
