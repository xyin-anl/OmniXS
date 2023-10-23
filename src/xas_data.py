from typing import TypedDict
import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


from utils.src.lightning.pl_data_module import PlDataModule


class XASData(PlDataModule):
    class DataQuery(TypedDict):
        compound: str
        simulation_type: Literal["VASP", "FEFF"]
        split: Literal["material", "spectra"]

    def __init__(
        self,
        query: "XASData.DataQuery",
        dtype: torch.dtype = torch.float32,
        **pl_data_module_kwargs,
    ):
        self.dtype = dtype
        self.query = query
        self.datasets = self.load_datasets(query=self.query, dtype=self.dtype)
        super().__init__(
            train_dataset=self.datasets["train"],
            val_dataset=self.datasets["val"],
            test_dataset=self.datasets["test"],
            **pl_data_module_kwargs,
        )

    @classmethod
    def load_datasets(
        self,
        query: DataQuery,
        dtype: torch.dtype = torch.float32,
    ) -> dict:
        np_data = self.load_data(query=query)
        return {
            task: TensorDataset(
                torch.from_numpy(np_data[task]["X"]).to(dtype=dtype),
                torch.from_numpy(np_data[task]["y"]).to(dtype=dtype),
            )
            for task in ["train", "val", "test"]
        }

    @classmethod
    def load_data(self, query: DataQuery) -> dict:
        data_dirs = self.get_data_dir(query)
        return {
            task: {
                "X": np.load(os.path.join(data_dirs, f"X_{task}.npy")),
                "y": np.load(os.path.join(data_dirs, f"y_{task}.npy")),
            }
            for task in ["train", "val", "test"]
        }

    @classmethod
    def get_data_dir(self, query: DataQuery) -> Union[str, Path]:
        return os.path.join(
            "dataset/ML-231009",
            f"{query['compound']}_K-edge_{query['simulation_type']}_XANES",
            f"{query['split']}-splits",
            "data",
        )


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
