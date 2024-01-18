import itertools
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Literal

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from src.data.material_split import MaterialSplitter
from utils.src.lightning.pl_data_module import PlDataModule


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        if not len(self.X) == len(self.y):
            raise ValueError("X and y must have same length")
        if self.X.dtype != self.y.dtype:
            raise ValueError("X and y must have same dtype")

    @property
    def tensors(self):
        return torch.tensor(self.X), torch.tensor(self.y)


@dataclass
class MLSplits:
    train: DataSplit
    val: DataSplit
    test: DataSplit

    @property
    def tensors(self):
        out_tensor = (self.train.tensors, self.val.tensors, self.test.tensors)
        flatten = itertools.chain.from_iterable
        return tuple(flatten(out_tensor))


@dataclass
class DataQuery:
    compound: str
    simulation_type: Literal["FEFF", "VASP"]


def load_xas_ml_data(
    query: DataQuery, split_fractions: List[float] = [0.8, 0.1, 0.1]
) -> MLSplits:
    """Loads data and does material splitting."""

    file_path = OmegaConf.load("./config/paths.yaml").paths.ml_data
    data_all = np.load(file_path.format(**query.__dict__))

    # greedy multiway partitioning
    idSite = list(zip(data_all["ids"], data_all["sites"]))
    train_idSites, val_idSites, test_idSites = MaterialSplitter.split(
        idSite=idSite, target_fractions=split_fractions
    )

    def to_split(ids):
        material_ids = ids[:, 0]
        sites = ids[:, 1]
        id_match = np.isin(data_all["ids"], material_ids)
        site_match = np.isin(data_all["sites"], sites)
        filter = np.where(id_match & site_match)[0]
        X = data_all["features"][filter].astype(np.float32)
        y = data_all["spectras"][filter].astype(np.float32)
        return DataSplit(X, y)

    return MLSplits(
        train=to_split(train_idSites),
        val=to_split(val_idSites),
        test=to_split(test_idSites),
    )



class XASPlData(PlDataModule):
    def __init__(
        self,
        query: DataQuery,
        dtype: torch.dtype = torch.float,
        split_fractions: List[float] = [0.8, 0.1, 0.1],
        **data_loader_kwargs,
    ):
        def dataset(split: DataSplit):
            X, y = split.tensors
            return TensorDataset(X.type(dtype), y.type(dtype))

        ml_split = load_xas_ml_data(query, split_fractions=split_fractions)
        super().__init__(
            train_dataset=dataset(ml_split.train),
            val_dataset=dataset(ml_split.val),
            test_dataset=dataset(ml_split.test),
            **data_loader_kwargs,
        )

if __name__ == "__main__":
    pl_data = XASPlData(query=DataQuery(compound="Cu", simulation_type="FEFF"))

    def print_fractions(xas_data):
        dataset = [xas_data.train_dataset, xas_data.val_dataset, xas_data.test_dataset]
        sizes = np.array([len(x) for x in dataset])
        sizes = sizes / sizes.sum()
        print(sizes)

    print_fractions(pl_data)  # [0.8 0.1 0.1]
