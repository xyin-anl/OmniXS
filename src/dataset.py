import os
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset


from utils.src.lightning.pl_data_module import PlDataModule


class XASDataModule(PlDataModule):
    def __init__(
        self,
        compound: Union[str, Path] = "Cu-O",
        split: Literal["material-splits", "random-splits"] = "material-splits",
        task: Literal["train", "val", "test"] = "train",
        dtype: torch.dtype = torch.float32,
        **pl_data_module_kwargs,
    ):
        self.compound = compound
        self.split = split
        self.task = task
        self.dtype = dtype
        train_dataset = self.np_data_to_dataset(
            *self.load_data(compound=compound, split=split, task="train")
        )
        val_dataset = self.np_data_to_dataset(
            *self.load_data(compound=compound, split=split, task="val")
        )
        test_dataset = self.np_data_to_dataset(
            *self.load_data(compound=compound, split=split, task="test")
        )
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **pl_data_module_kwargs,
        )

    def np_data_to_dataset(self, x, y, device="cpu"):
        return TensorDataset(
            torch.from_numpy(x).to(dtype=self.dtype),
            torch.from_numpy(y).to(dtype=self.dtype),
        )

    @classmethod
    def load_data(
        self,
        data_dir: Union[str, Path] = "dataset/ML-231009",
        compound: Union[str, Path] = "Cu-O",
        simulation_type: Literal["FEFF", "VASP"] = "FEFF",
        split: Literal["material-splits", "random-splits"] = "material-splits",
        task: Literal["train", "val", "test"] = "train",
    ):
        data_compound_dir = os.path.join(
            data_dir, compound + "_K-edge_" + simulation_type + "_XANES", split, "data"
        )
        x = np.load(os.path.join(data_compound_dir, f"X_{task}.npy"))
        y = np.load(os.path.join(data_compound_dir, f"y_{task}.npy"))
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_{task}.npy and y_{task}.npy have different lengths \
                for compound {compound} and split_type {split}"
            )
        return x, y


"dataset/ML-231009/Cu-O-edge_FEFF_XANES/material-splits/data"

if __name__ == "__main__":
    from utils.src.plots.heatmap_of_lines import heatmap_of_lines
    import matplotlib.pyplot as plt

    compounds = ["Cu-O", "Ti-O"]
    tasks = ["train", "test"]
    simulation_type = "FEFF"
    for idx, name in [(1, "spectra"), (0, "features")]:
        for compound in compounds:
            for task in tasks:
                data = XASDataModule.load_data(compound=compound, task=task)[idx]
                title = f"{name}_{compound}_{simulation_type}_{task}"
                # features_Cu-O_FEFF_train_Cu-O_FEFF_test_Ti-O_FEFF_train_Ti-O_FEFF_test.png
                heatmap_of_lines(data=data, title=title)
                plt.savefig(f"{title}.png", dpi=300)
