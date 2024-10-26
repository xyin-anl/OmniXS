# %%

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, List, Literal, Optional, Self, Union

# from get_eta import get_eta
from refactor.data.merge_ml_splits import FEFFSplits, MergedSplits

import numpy as np
import torch
from lightning.pytorch import Trainer
from matplotlib import pyplot as plt
from pydantic import BaseModel, computed_field
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from refactor.data import (
    FEFF,
    VASP,
    DataTag,
    IdentityScaler,
    ScaledMlSplit,
    ThousandScaler,
    Ti,
    UncenteredRobustScaler,
    FEFFDataTags,
    VASPDataTags,
)
from refactor.model.trained_model import ModelTag, TrainedModelLoader
from refactor.model.training import LightningXASData
from refactor.utils.io import DEFAULTFILEHANDLER


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


class TrainedXASBlock(TrainedModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        return self.model(X.to(self.model.device)).detach().cpu().numpy()

    @classmethod
    def load(
        cls,
        tag: ModelTag,
        train_x_scaler: type = ThousandScaler,
        train_y_scaler: type = ThousandScaler,
    ):
        model = TrainedModelLoader.load_model(tag)
        model.eval()
        model.freeze()
        instance = cls(
            tag=tag,
            model=model,
            train_x_scaler=train_x_scaler,
            train_y_scaler=train_y_scaler,
        )
        return instance


class MeanModel(TrainedModel):

    @computed_field
    def train_mean(self) -> np.ndarray:
        return self.default_split.train.y.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.repeat(self.train_mean[None, :], len(X), axis=0)


def get_eta(model_tag: ModelTag):
    x_scaler = ThousandScaler
    y_scaler = ThousandScaler
    scalers = dict(train_x_scaler=x_scaler, train_y_scaler=y_scaler)
    model = TrainedXASBlock.load(model_tag, **scalers)
    mean = MeanModel(tag=model_tag, **scalers)
    model_mse = model.metrics.median_of_mse_per_spectra
    mean_mse = mean.metrics.median_of_mse_per_spectra
    return mean_mse / model_mse


# %%

EXPERTFEFFS = [
    ModelTag(name="expertXAS", **data_tag.dict()) for data_tag in FEFFDataTags
]
EXPERTVASPS = [
    ModelTag(name="expertXAS", **data_tag.dict()) for data_tag in VASPDataTags
]
EXPERTXASTAGS = EXPERTFEFFS + EXPERTVASPS
TUNEDUNIVERSALXASTAGS = [
    ModelTag(name="tunedUniversalXAS", element=tag.element, type=tag.type)
    for tag in EXPERTXASTAGS
]


eta_values = {}
for set_name, tag_set in zip(
    ["expertXAS", "tunedUniversalXAS"],
    [EXPERTXASTAGS, TUNEDUNIVERSALXASTAGS],
):
    eta_set = {}
    for model_tag in tag_set:
        if (
            model_tag.element == "Ti"
            and model_tag.type == "VASP"
            and set_name == "tunedUniversalXAS"
        ):
            print("Skipping Ti VASP with dummy")
            eta = np.nan
        else:
            eta = get_eta(model_tag)
        print(f"{model_tag.element}_{model_tag.type}: {eta}")
        eta_set[f"{model_tag.element}_{model_tag.type}"] = eta
    eta_values[set_name] = eta_set

# %%

for element in eta_values["expertXAS"].keys():
    expert = eta_values["expertXAS"][element]
    tuned = eta_values["tunedUniversalXAS"][element]
    print(
        f"{element}: \t{expert:.2f},\t{tuned:.2f} \t({(tuned-expert)/expert*100:.2f}%)"
    )


# %%

tag = ModelTag(name="expertXAS", element="Ti", type="VASP")

model = TrainedXASBlock.load(
    tag,
    train_x_scaler=ThousandScaler,
    train_y_scaler=ThousandScaler,
)

mean_model = MeanModel(
    tag=tag,
    train_x_scaler=ThousandScaler,
    train_y_scaler=ThousandScaler,
)


# %%
plt.scatter(
    mean_model.metrics.targets,
    mean_model.metrics.predictions,
    s=1,
    alpha=0.5,
    color="blue",
    label="Mean",
)

plt.scatter(
    model.metrics.targets,
    model.metrics.predictions,
    s=1,
    alpha=0.5,
    color="red",
    label="Expert",
)

plt.gca().set_aspect("equal", adjustable="box")
plt.gca().set_xlabel("True")
plt.gca().set_ylabel("Predicted")
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color="black")
plt.legend()

# %%

plt.plot(model.metrics.targets.T, label="True", color="green", linestyle="-.")
plt.plot(model.metrics.predictions.T, label="Expert", color="red")
plt.plot(mean_model.metrics.predictions.T, label="Mean")

# %%


idx = np.random.randint(0, len(model.metrics.predictions))
plt.plot(model.metrics.predictions[idx], label="Expert", color="gray", alpha=0.5)
plt.plot(mean_model.metrics.predictions[idx], label="Mean")
plt.plot(model.metrics.targets[idx], label="True", color="green", linestyle="--")
# fill from predictions to true
plt.fill_between(
    np.arange(len(model.metrics.predictions[idx])),
    model.metrics.predictions[idx],
    model.metrics.targets[idx],
    color="gray",
    alpha=0.5,
)
plt.legend()


# %%

universal_tag = ModelTag(
    name="universalXAS",
    element="All",
    type="FEFF",
)
universalXAS = TrainedXASBlock.load(
    tag=universal_tag,
    train_x_scaler=ThousandScaler,
    train_y_scaler=ThousandScaler,
)
merged_feff_splits = MergedSplits.load(FEFFDataTags, DEFAULTFILEHANDLER, balanced=False)


# %%


def get_universal_model_eta(data_tag: DataTag):
    # split = merged_feff_splits.splits[data_tag]
    split = TrainedModelLoader.load_ml_splits(data_tag)
    split = ScaledMlSplit.from_splits(
        split,
        x_scaler=ThousandScaler,
        y_scaler=ThousandScaler,
    )
    universal_element_metric = universalXAS.compute_metrics(
        split
    ).median_of_mse_per_spectra
    expert_element_metric = MeanModel(
        # tag=ModelTag(name="expertXAS", **FEFFDataTags[0].dict()),
        tag=ModelTag(name="expertXAS", **data_tag.dict()),
        train_x_scaler=ThousandScaler,
        train_y_scaler=ThousandScaler,
    ).metrics.median_of_mse_per_spectra
    return expert_element_metric / universal_element_metric


universal_metrics = {}
for data_tag in FEFFDataTags:
    eta = get_universal_model_eta(data_tag)
    universal_metrics[data_tag.element + data_tag.type] = eta
    print(f"{data_tag.element}: {eta}")

# %%

len(merged_feff_splits)

# %%
