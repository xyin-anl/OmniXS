# %%

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, List, Literal, Optional, Self, Union

import numpy as np
import torch
from lightning.pytorch import Trainer
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, computed_field
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from refactor.data import (
    FEFF,
    VASP,
    Cu,
    DataTag,
    IdentityScaler,
    MLData,
    MLSplits,
    ScaledMlSplit,
    ThousandScaler,
    Ti,
    V,
    Cr,
    Mn,
    UncenteredRobustScaler,
    FEFFDataTags,
    VASPDataTags,
)
from refactor.model.training import LightningXASData, PlModule
from refactor.model.xasblock import XASBlock
from refactor.utils.io import DEFAULTFILEHANDLER, FileHandler


class ModelTag(DataTag):
    name: str = Field(..., description="Name of the model")


class TrainedModelLoader:
    file_handler: FileHandler = DEFAULTFILEHANDLER
    hparams = OmegaConf.load("config/training/hparams.yaml").hparams
    input_dim = 64  # TODO: remove hardcoding of i/o dims
    output_dim = 141

    @staticmethod
    def load_model(tag: ModelTag):
        return PlModule.load_from_checkpoint(
            checkpoint_path=TrainedModelLoader.get_ckpt_path(tag),
            model=XASBlock(**TrainedModelLoader.get_layer_widths(tag)),
        )

    @staticmethod
    def get_layer_widths(tag: ModelTag):
        hidden_widths = TrainedModelLoader.hparams[tag.name][tag.type][
            tag.element
        ].widths
        return dict(
            input_dim=TrainedModelLoader.input_dim,
            hidden_dims=hidden_widths,
            output_dim=TrainedModelLoader.output_dim,
        )

    @staticmethod
    def get_ckpt_path(tag: ModelTag):
        paths = list(
            TrainedModelLoader.file_handler.serialized_objects_filepaths(
                "TrainedXASBlock", **tag.dict()
            )
        )
        if len(paths) != 1:
            raise ValueError(f"Expected 1 path, got {len(paths)}")
        return paths[0]

    @staticmethod
    def load_ml_splits(tag: ModelTag):
        return TrainedModelLoader.file_handler.deserialize_json(
            MLSplits,
            DataTag(element=tag.element, type=tag.type),
        )

    @staticmethod
    def load_scaled_splits(tag: ModelTag, scaler: type = RobustScaler):
        splits = TrainedModelLoader.load_ml_splits(tag)
        return ScaledMlSplit(x_scaler=scaler(), y_scaler=scaler(), **splits.dict())


class ModelMetrics(BaseModel):
    predictions: np.ndarray
    targets: np.ndarray

    @computed_field
    def mse(self) -> float:
        return mean_squared_error(self.targets, self.predictions)

    @computed_field
    def mse_per_spectra(self) -> np.ndarray:
        return np.mean((self.targets - self.predictions) ** 2, axis=1)

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
    def deciles(self):
        return self.top_predictions(splits=10, drop_last_split=True)

    class Config:
        arbitrary_types_allowed = True


class TrainedModel(BaseModel, ABC):
    tag: ModelTag
    model: Optional[Any] = None
    train_scaler: type = ThousandScaler

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @computed_field
    def metrics(self) -> ModelMetrics:
        return self.compute_metrics(self.default_split)

    @property
    def default_split(self) -> ScaledMlSplit:
        return TrainedModelLoader.load_scaled_splits(self.tag, self.train_scaler)

    def compute_metrics(self, splits: ScaledMlSplit) -> ModelMetrics:
        predictions = self.predict(splits.test.X)
        targets = splits.test.y
        predictions = splits.y_scaler.inverse_transform(predictions)
        targets = splits.y_scaler.inverse_transform(targets)
        return ModelMetrics(targets=targets, predictions=predictions)

    class Config:
        arbitrary_types_allowed = True


class TrainedXASBlock(TrainedModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        return self.model(X.to(self.model.device)).detach().cpu().numpy()

    @classmethod
    def load(cls, tag: ModelTag):
        model = TrainedModelLoader.load_model(tag)
        model.eval()
        model.freeze()
        instance = cls(tag=tag, model=model, train_scaler=ThousandScaler)
        instance.model_rebuild()  # Explicitly rebuild the model
        return instance


class MeanModel(TrainedModel):

    @computed_field
    def train_mean(self) -> np.ndarray:
        return self.default_split.train.y.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(self.train_mean[None, :], len(X), axis=0)


EXPERTFEFFS = [
    ModelTag(name="expertXAS", **data_tag.dict()) for data_tag in FEFFDataTags
]
EXPERTVASPS = [
    ModelTag(name="expertXAS", **data_tag.dict()) for data_tag in VASPDataTags
]
EXPERTXASTAGS = EXPERTFEFFS + EXPERTVASPS


def get_eta(model_tag: ModelTag):
    expert = TrainedXASBlock.load(model_tag)
    mean = MeanModel(tag=model_tag)
    return (
        mean.metrics.median_of_mse_per_spectra
        / expert.metrics.median_of_mse_per_spectra
    )


# %%

for model_tag in EXPERTXASTAGS:
    print(f"{model_tag.element}_{model_tag.type}: {get_eta(model_tag)}")


# %%

expert = TrainedXASBlock.load(EXPERTXASTAGS[0])
mean = MeanModel(tag=EXPERTXASTAGS[0])
plt.plot(expert.metrics.targets.T, c="gray", alpha=0.1)
plt.plot(expert.metrics.predictions.T, c="blue", linestyle="-.")
plt.plot(mean.metrics.predictions.T, c="red", linestyle="--")

# %%

idx = np.random.randint(0, len(expert.metrics.targets), 1)[0]
plt.plot(expert.metrics.targets[idx], c="gray", label="Target")
plt.plot(expert.metrics.predictions[idx], c="blue", linestyle="-.", label="Expert")
plt.legend()

# %%

for d in expert.metrics.deciles:
    plt.plot(d[0], c="blue", linestyle="-.")
    plt.plot(d[1], c="red", linestyle="-")
    plt.show()


# %%

MeanModel(tag=EXPERTVASPS[0]).metrics.mse, TrainedXASBlock.load(
    EXPERTVASPS[0]
).metrics.mse, get_eta(EXPERTVASPS[0])

# %%
