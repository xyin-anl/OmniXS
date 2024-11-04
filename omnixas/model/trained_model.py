# %%
from abc import ABC, abstractmethod
from typing import Any, Optional, Self

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, computed_field
from sklearn.metrics import mean_squared_error
from torch import nn

from omnixas.data import (
    DataTag,
    IdentityScaler,
    MLSplits,
    ScaledMlSplit,
    ThousandScaler,
)
from omnixas.data.merge_ml_splits import FEFFSplits
from omnixas.model.training import PlModule
from omnixas.model.xasblock import XASBlock
from omnixas.utils.io import DEFAULTFILEHANDLER, FileHandler


class ModelTag(DataTag):
    name: str = Field(..., description="Name of the model")


class ModelMetrics(BaseModel):
    predictions: np.ndarray
    targets: np.ndarray

    @computed_field
    def mse(self) -> float:
        return mean_squared_error(self.targets, self.predictions)

    @property
    def median_of_mse_per_spectra(self) -> float:
        return np.median(self.mse_per_spectra)

    def _sorted_predictions(self, sort_array=None):
        sort_array = self.mse_per_spectra
        pair = np.column_stack((self.targets, self.predictions))
        pair = pair.reshape(-1, 2, self.targets.shape[1])
        pair = pair[sort_array.argsort()]
        return pair, sort_array.argsort()

    def top_predictions(self, splits=10, drop_last_split=True):
        pair = self._sorted_predictions()[0]
        new_len = len(pair) - divmod(len(pair), splits)[1]
        pair = pair[:new_len]
        top_spectra = [s[-1] for s in np.split(pair, splits)]  # bottom of each split
        if drop_last_split:
            top_spectra = top_spectra[:-1]
        return np.array(top_spectra)

    @property
    def residuals(self):
        return self.targets - self.predictions

    @property
    def mse_per_spectra(self):
        return np.mean(self.residuals**2, axis=1)

    @property
    def deciles(self):
        return self.top_predictions(splits=10, drop_last_split=True)

    class Config:
        arbitrary_types_allowed = True


class ComparisonMetrics(BaseModel):
    metric1: ModelMetrics = Field(..., description="First metric to compare")
    metric2: ModelMetrics = Field(..., description="Second metric to compare")

    @property
    def residual_diff(self):
        return np.abs(self.metric1.residuals) - np.abs(self.metric2.residuals)


class TrainedModel(BaseModel, ABC):
    tag: ModelTag
    model: Optional[Any] = None
    train_x_scaler: type = ThousandScaler
    train_y_scaler: type = ThousandScaler

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @computed_field
    def metrics(self) -> ModelMetrics:
        return self.compute_metrics(self.default_split)

    @property
    def default_split(self) -> ScaledMlSplit:
        if self.tag.element == "All":
            return FEFFSplits()
        return TrainedModelLoader.load_scaled_splits(
            self.tag,
            self.train_x_scaler,
            self.train_y_scaler,
        )

    def compute_metrics(self, splits: ScaledMlSplit) -> ModelMetrics:
        predictions = self.predict(splits.test.X)
        targets = splits.test.y
        predictions = splits.y_scaler.inverse_transform(predictions)
        targets = splits.y_scaler.inverse_transform(targets)
        return ModelMetrics(targets=targets, predictions=predictions)

    class Config:
        arbitrary_types_allowed = True


class MeanModel(TrainedModel):

    @classmethod
    def from_data_tag(cls, data_tag: DataTag, **kwargs) -> Self:
        model_tag = ModelTag(name="meanModel", **data_tag.dict())
        return cls(tag=model_tag, **kwargs)

    @computed_field
    def train_mean(self) -> np.ndarray:
        return self.default_split.train.y.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(self.train_mean[None, :], len(X), axis=0)


class TrainedXASBlock(TrainedModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        return self.model(X.to(self.model.device)).detach().cpu().numpy()

    @classmethod
    def load(
        cls,
        tag: ModelTag,
        x_scaler: type = ThousandScaler,
        y_scaler: type = ThousandScaler,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ):
        model = TrainedModelLoader.load_model(tag, file_handler)
        model.eval()
        model.freeze()
        instance = cls(
            tag=tag,
            model=model,
            train_x_scaler=x_scaler,
            train_y_scaler=y_scaler,
        )
        return instance


class TrainedModelLoader:

    @staticmethod
    def load_model_for_finetuning(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
        freeze_first_k_layers: int = 0,
    ) -> PlModule:
        model = TrainedModelLoader.load_model(tag, file_handler)

        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        layer_count = 0
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                if layer_count < freeze_first_k_layers:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_running_stats()
                if isinstance(m, nn.Linear):
                    layer_count += 1
        return model

    @staticmethod
    def load_model(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ) -> PlModule:
        ckpt_path = TrainedModelLoader.get_ckpt_path(tag, file_handler)
        print(f"Loading model from {ckpt_path}")
        return PlModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=XASBlock(**TrainedModelLoader.get_layer_widths(tag)),
        )

    @staticmethod
    def get_layer_widths(
        tag: ModelTag,
        hparams: dict = OmegaConf.load("config/training/hparams.yaml").hparams,
        input_dim: int = 64,
        output_dim: int = 141,
    ):
        if tag.name != "tunedUniversalXAS":
            hidden_widths = hparams[tag.name][tag.type][tag.element].widths
        else:
            hidden_widths = hparams["universalXAS"]["FEFF"]["All"].widths  # TODO: hacky

        return dict(
            input_dim=input_dim,
            hidden_dims=hidden_widths,
            output_dim=output_dim,
        )

    @staticmethod
    def get_ckpt_path(
        tag: ModelTag,
        file_handler=DEFAULTFILEHANDLER(),
    ):
        paths = file_handler.serialized_objects_filepaths(
            "TrainedXASBlock", **tag.dict()
        )
        if len(paths) != 1:
            raise ValueError(f"Expected 1 path, got {len(paths)}")
        return paths[0]

    @staticmethod
    def load_ml_splits(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ):
        return file_handler.deserialize_json(
            MLSplits,
            DataTag(element=tag.element, type=tag.type),
        )

    @staticmethod
    def load_scaled_splits(
        tag: ModelTag,
        x_scaler: type = IdentityScaler,
        y_scaler: type = ThousandScaler,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ):
        splits = TrainedModelLoader.load_ml_splits(tag, file_handler)
        return ScaledMlSplit(
            x_scaler=x_scaler(),
            y_scaler=y_scaler(),
            **splits.dict(),
        )


# %%
