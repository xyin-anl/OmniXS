import os
import warnings
from typing import List, Literal, TypedDict

import numpy as np
import torch
from torch.utils.data import TensorDataset

from config.defaults import cfg
from src.data.material_split import MaterialSplitter
from utils.src.lightning.pl_data_module import PlDataModule


class XASData(PlDataModule):
    class DataQuery(TypedDict):
        compound: str
        simulation_type: Literal["VASP", "FEFF"]

    def __init__(
        self,
        query: "XASData.DataQuery",
        dtype: torch.dtype = torch.float32,
        pre_split: bool = False,
        split_fractions: List[float] = [0.8, 0.1, 0.1],  #  used when pre_split=False
        max_size: int = None,
        **pl_data_module_kwargs,
    ):
        self.dtype = dtype
        self.query = query

        if pre_split and split_fractions is not None:
            raise ValueError(
                "split_fractions must be None when pre_split is set to True"
            )
        if max_size is not None and pre_split:
            raise ValueError("max_size must be None when pre_split is set to True")

        if pre_split:
            self.datasets = XASData.load_pre_split_data(self.query, self.dtype)
        else:
            self.datasets = self.load_data(
                query=self.query,
                dtype=self.dtype,
                split_fractions=split_fractions,
                max_size=max_size,
            )

        super().__init__(
            train_dataset=self.datasets["train"],
            val_dataset=self.datasets["val"],
            test_dataset=self.datasets["test"],
            **pl_data_module_kwargs,
        )

    @staticmethod
    def load_data(
        query: DataQuery,
        dtype: torch.dtype = torch.float32,
        split_fractions: List[float] = [0.8, 0.1, 0.1],
        max_size: int = None,
    ):
        """Loads data and does material splitting."""
        compound = query["compound"]
        sim_type = query["simulation_type"]
        file_path = cfg.paths.ml_data.format(
            compound=compound, simulation_type=sim_type
        )

        data_all = np.load(file_path)

        # downsampling
        if max_size is not None:
            assert max_size <= len(data_all["ids"]), "max_size must be <= len(data)"
            idx = np.random.choice(len(data_all["ids"]), max_size, replace=False)
            data_all = {
                k: v[idx] for k, v in data_all.items() if k != "energies"
            }  # coz energies is a one dimensional array

        id_site_pairs = list(zip(data_all["ids"], data_all["sites"]))

        # uses greedy multiway partitioning
        train_split, val_split, test_split = MaterialSplitter.split(
            id_site_pairs=id_site_pairs,
            target_fractions=split_fractions,
        )

        # used later to map ids to tasks
        id_to_task = {id: "train" for id in train_split[:, 0]}
        id_to_task.update({id: "val" for id in val_split[:, 0]})
        id_to_task.update({id: "test" for id in test_split[:, 0]})

        # Initialize split data structure
        variables = ["X", "y"]
        tasks = ["train", "val", "test"]
        data_split = {t: {v: [] for v in variables} for t in tasks}

        # Populate split data
        for id, feature, spectra in zip(
            data_all["ids"], data_all["features"], data_all["spectras"]
        ):
            task = id_to_task.get(id)
            if task:
                data_split[task]["X"].append(feature)
                data_split[task]["y"].append(spectra)
            else:
                warnings.warn(f"ID {id} not in any recognized split")

        # Convert to tensors and create TensorDataset objects
        for t in tasks:
            data_split[t] = {
                v: torch.from_numpy(np.array(data_split[t][v])).to(dtype=dtype)
                for v in variables
            }
            data_split[t] = TensorDataset(data_split[t]["X"], data_split[t]["y"])

        return data_split

    @staticmethod
    def load_pre_split_data(query: DataQuery, dtype: torch.dtype = torch.float32):
        """Legacy method. Loads data that has already been split."""
        compound = query["compound"]
        sim_type = query["simulation_type"]
        data_dir = cfg.paths.pre_split_ml_data
        data_dir = data_dir.format(compound=compound, simulation_type=sim_type)

        def load_split(task, var_name):
            return np.load(os.path.join(data_dir, f"{var_name}_{task}.npy"))

        def to_tensor(data):
            return torch.from_numpy(data).to(dtype=dtype)

        def to_dataset(X, y):
            return TensorDataset(to_tensor(X), to_tensor(y))

        tasks = ["train", "val", "test"]
        return {t: to_dataset(load_split(t, "X"), load_split(t, "y")) for t in tasks}


if __name__ == "__main__":
    xas_data_pre_split = XASData(
        query=XASData.DataQuery(compound="Cu-O", simulation_type="FEFF"),
        pre_split=True,
        split_fractions=None,
    )

    xas_data_post_split = XASData(
        query=XASData.DataQuery(compound="Cu", simulation_type="FEFF"),
        pre_split=False,
        split_fractions=[0.4, 0.3, 0.3],
        max_size=1000,
    )

    def print_fractions(xas_data):
        sizes = np.array([len(x) for x in xas_data.datasets.values()])
        sizes = sizes / sizes.sum()
        print(sizes)

    print_fractions(xas_data_pre_split)  # [0.8 0.1 0.1]
    print(len(xas_data_pre_split.datasets["train"]))  # ~ 400

    print_fractions(xas_data_post_split)  # [0.4 0.3 0.3]
    print(len(xas_data_post_split.datasets["train"]))  # ~ 400
