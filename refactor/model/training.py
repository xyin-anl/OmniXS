# %%

import numpy as np
from typing import Callable, Optional

import hydra
import lightning
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset

from refactor.data import DataTag, MLData, MLSplits
from refactor.data.scaler import ScaledMlSplit
from refactor.utils.callbacks import TensorboardLogTestTrainLoss
from refactor.utils.io import DEFAULTFILEHANDLER, FileHandler
from refactor.data.scaler import ThousandScaler, IdentityScaler, UncenteredRobustScaler

# %%


class LightningXASData(lightning.LightningDataModule):
    def __init__(
        self,
        ml_splits: MLSplits,
        batch_size: int,
        scaler: type = ThousandScaler,
        **kwargs,
    ):
        super().__init__()
        self.splits = ml_splits
        self.batch_size = batch_size
        self.scaler = scaler
        self.kwargs = kwargs

    @staticmethod
    def to_tensor(arr: np.ndarray):
        # used by external classes for consistent dtype
        return tensor(arr, dtype=torch.float32)

    @staticmethod
    def to_tensor_dataset(split: MLData):
        return TensorDataset(
            LightningXASData.to_tensor(split.X),
            LightningXASData.to_tensor(split.y),
        )

    def setup(self, stage: str = None):
        scaled_splits = ScaledMlSplit(
            x_scaler=self.scaler(),
            y_scaler=self.scaler(),
            **self.splits.dict(),
        )
        self.train = self.to_tensor_dataset(scaled_splits.train)
        self.val = self.to_tensor_dataset(scaled_splits.val)
        self.test = self.to_tensor_dataset(scaled_splits.test)
        return self

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, **self.kwargs)


class PlModule(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_metric: Callable[[], torch.nn.Module] = torch.nn.MSELoss,
        optimizer: Callable = torch.optim.Adam,
        lr: Optional[float] = 0.0001,
    ):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.loss = loss_metric()
        self.model = model

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def forward(self, x):
        return self.model(x)

    def logged_loss(self, name, y, y_pred):
        loss = self.loss(y, y_pred)
        self.log(name, loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.logged_loss("train_loss", y, self.model(x))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self.logged_loss("val_loss", y, self.model(x))

    def test_step(self, batch, batch_idx):
        x, y = batch
        return self.logged_loss("test_loss", y, self.model(x))

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.model(x)


@hydra.main(version_base=None)
def trainModel(cfg: DictConfig):
    # torch.set_float32_matmul_precision('medium') # useful for tensor cores
    data_module = instantiate(cfg.data_module)
    module = instantiate(cfg.module)
    trainer = instantiate(cfg.trainer)
    trainer.callbacks.extend([TensorboardLogTestTrainLoss()])
    trainer.fit(module, data_module)
    trainer.test(module, datamodule=data_module)


# %%


if __name__ == "__main__":

    trainModel()

    # from refactor.data import FEFF, Cu
    # from refactor.model.xasblock import XASBlock
    # tag = DataTag(element=Cu, type=FEFF)
    # datamodule = LightningXASData(tag, 128).setup()
    # print(f"Train: {len(datamodule.train_dataloader())}")
    # print(f"Val: {len(datamodule.val_dataloader())}")
    # print(f"Test: {len(datamodule.test_dataloader())}")
    # pl_model = LightningXASBlock(dims=[64, 100, 141])
    # print("PL Model: ", pl_model)
    # trainer = lightning.Trainer(max_epochs=1)
    # trainer.fit(pl_model, datamodule)
