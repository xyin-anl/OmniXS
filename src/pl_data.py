from config.defaults import cfg
from typing import TypedDict
import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


from utils.src.lightning.pl_data_module import PlDataModule


class DataQuery(TypedDict):
    compound: str
    simulation_type: Literal["VASP", "FEFF"]


class XASData(PlDataModule):
    def __init__(
        self,
        query: "XASData.DataQuery",
        dtype: torch.dtype = torch.float32,
        pre_split: bool = False,
        **pl_data_module_kwargs,
    ):
        self.dtype = dtype
        self.query = query
        if pre_split:
            self.datasets = XASData.load_pre_split_data(self.query, self.dtype)
        else:
            self.datasets = self.load_datasets(query=self.query, dtype=self.dtype)

        super().__init__(
            train_dataset=self.datasets["train"],
            val_dataset=self.datasets["val"],
            test_dataset=self.datasets["test"],
            **pl_data_module_kwargs,
        )

    @staticmethod
    def load_pre_split_data(query: DataQuery, dtype: torch.dtype = torch.float32):
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
    from utils.src.plots.heatmap_of_lines import heatmap_of_lines
    import matplotlib.pyplot as plt

    for plot_name in ["features", "spectra"]:
        for compound in ["Cu-O", "Ti-O"]:
            for task in ["train", "test"]:
                query = {
                    "compound": compound,
                    "simulation_type": "FEFF",
                    "split": "material",
                    "task": task,
                }
                var_name = "X" if plot_name == "features" else "y"
                data = XASData.load_data(query)[task][var_name]
                title = f"{plot_name}_{compound}_{query['simulation_type']}_{task}"
                heatmap_of_lines(data=data, title=title)
                plt.savefig(f"{title}.png", dpi=300)
