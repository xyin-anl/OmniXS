from functools import cached_property
import pickle
from typing import Union
import os
import itertools
from dataclasses import dataclass
from typing import Any, List, Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from config.defaults import cfg

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


class FeatureProcessor:
    def __init__(self, query: DataQuery, data_splits: MLSplits = None):
        self.query = query
        # none option set to access saved pca and scaler from cache
        self.data_splits = data_splits

    @cached_property
    def splits(self):
        return MLSplits(
            train=self._reduce_feature_dims(self.data_splits.train),
            val=self._reduce_feature_dims(self.data_splits.val),
            test=self._reduce_feature_dims(self.data_splits.test),
        )

    def _reduce_feature_dims(self, data_splits: DataSplit):
        return DataSplit(
            self.pca.transform(self.scaler.transform(data_splits.X)), data_splits.y
        )

    @cached_property
    def scaler(self):
        # load from cache if cache exists
        scaler_cache_path = cfg.paths.cache.scaler.format(**self.query.__dict__)
        if os.path.exists(scaler_cache_path):  # load from cache
            with open(scaler_cache_path, "rb") as f:
                return pickle.load(f)
        # else fit scaler and save to cache
        if self.data_splits is None:
            raise ValueError("data_splits is None. Cannot fit scaler.")
        scaler = StandardScaler().fit(self.data_splits.train.X)
        os.makedirs(os.path.dirname(scaler_cache_path), exist_ok=True)
        with open(scaler_cache_path, "wb") as f:
            pickle.dump(scaler, f)
        return scaler

    @cached_property
    def pca(self):
        # load from cache if cache exists
        pca_cache_path = cfg.paths.cache.pca.format(**self.query.__dict__)
        if os.path.exists(pca_cache_path):
            with open(pca_cache_path, "rb") as f:
                return pickle.load(f)
        # else fit pca and save to cache
        if self.data_splits is None:
            raise ValueError("data_splits is None. Cannot fit pca.")
        pca = PCA(n_components=cfg.dscribe.pca.n_components)
        pca.fit(self.data_splits.train.X)
        os.makedirs(os.path.dirname(pca_cache_path), exist_ok=True)
        with open(pca_cache_path, "wb") as f:
            pickle.dump(pca, f)
        return pca

    def _test_if_pca_matches_config(self):
        # expected number of components or the explained variance
        expected_pca_param = cfg.dscribe.pca.n_components_
        if not self.pca.n_components == expected_pca_param:
            msg = "PCA components mismatch: "
            msg += f"{self.pca.n_components_} != {expected_pca_param} for {self.query}"
            raise ValueError(msg)


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
        X *= 1000
        y *= 1000
        return DataSplit(X, y)

    return MLSplits(
        train=to_split(train_idSites),
        val=to_split(val_idSites),
        test=to_split(test_idSites),
    )

    if reduce_dims is None:
        reduce_dims = query.simulation_type in cfg.dscribe.features
    return FeatureProcessor(query, splits).splits if reduce_dims else splits


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

    # =============================================================================
    # tests if pca and scaler are cached
    # =============================================================================
    from p_tqdm import p_map

    load_xas_ml_data(DataQuery("Cu", "SOAP"))

    # # should cache pca and scaler
    # p_map(
    #     lambda q: load_xas_ml_data(q),
    #     [DataQuery(c, "SOAP") for c in cfg.compounds[-1]],
    #     num_cpus=1,
    # )

    # for compound in cfg.compounds:
    #     query = DataQuery(compound=compound, simulation_type="SOAP")
    #     # caching pca
    #     ml_split = load_xas_ml_data(query)
    #     pca = FeatureProcessor(query).pca  # should load from cache
    #     print(f"PCA components: {pca.n_components_} for {query}")

    # =============================================================================
    print("dummy")

    # pl_data = XASPlData(query=DataQuery(compound="Cu", simulation_type="FEFF"))
    # def print_fractions(xas_data):
    #     dataset = [xas_data.train_dataset, xas_data.val_dataset, xas_data.test_dataset]
    #     sizes = np.array([len(x) for x in dataset])
    #     sizes = sizes / sizes.sum()
    #     print(sizes)
    # print_fractions(pl_data)  # [0.8 0.1 0.1]
