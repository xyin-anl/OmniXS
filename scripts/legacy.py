from src.data.material_split import MaterialSplitter
from src.data.ml_data import DataSplit, MLSplits
from src.data.ml_data import DataQuery
import torch
from config.defaults import cfg
import numpy as np
import os


def load_pre_split_data(query: DataQuery):
    """Legacy method. Loads data that has already been split."""
    dir = cfg.paths.pre_split_ml_data
    dir = dir.format(compound=query.compound, simulation_type=query.simulation_type)

    def load_split(task, var_name):
        return np.load(os.path.join(dir, f"{var_name}_{task}.npy")).astype(np.float32)

    return MLSplits(
        train=DataSplit(load_split("train", "X"), load_split("train", "y")),
        val=DataSplit(load_split("val", "X"), load_split("val", "y")),
        test=DataSplit(load_split("test", "X"), load_split("test", "y")),
    )
