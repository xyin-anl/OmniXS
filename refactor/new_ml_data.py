# %%
from refactor.spectra_enums import Element, SpectrumType
from refactor.spectra_data import (
    AbsorptionSpectra,
    Spectrum,
    Material,
    AbsorptionSite,
)
import json
from box import Box  # converts dict keys to attributes (AttributeDict)
from enum import StrEnum
from typing import Dict
from pydantic import (
    BaseModel,
    PositiveInt,
    Field,
    validator,
    NonNegativeFloat,
    NonNegativeInt,
    StrictInt,
    field_validator,
    model_validator,
    field_serializer,
    model_serializer,
    computed_field,
)
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Self, Union, Optional
from pathlib import Path
from refactor.spectra_enums import StrEnum

import numpy as np
from matgl.ext.pymatgen import Structure

from refactor.utilities.featurizer.m3gnet_featurizer import MaterialFeatureType


# %%


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    NONE = "none"


class XASBlockTypes(StrEnum):
    EXPERT = "expert"
    UNIVERSAL = "universal"
    TUNED = "tuned"


# =========================================


class MLPConfig(BaseModel):
    input_dim: PositiveInt
    output_dim: PositiveInt
    hidden_dims: List[PositiveInt] = Field(..., min_length=1)


class TrainConfig(BaseModel):
    epochs: PositiveInt = Field(default=100)
    batch_size: PositiveInt = Field(default=32)
    learning_rate: float = Field(..., gt=0, lt=1)


class XASMlDataConfig(BaseModel):
    feature: MaterialFeatureType
    elements: List[Element] = Field(..., min_length=1, set=True)
    split_fractions: List[float] = Field(..., min_length=3, max_length=3)

    @validator("split_fractions")
    def check_split_fractions(cls, v):
        if sum(v) != 1:
            raise ValueError("Split fractions must sum to 1")
        return v


class XASBlockConfig(BaseModel):
    type: XASBlockTypes
    mlp: MLPConfig
    train: TrainConfig
    data: XASMlDataConfig


class MaterialFeaturizer(ABC):

    @abstractmethod
    def featurize(self, structure: Structure) -> Any:
        raise NotImplementedError


class MaterialFeatureFactory:

    @staticmethod
    def featurizer(feature: MaterialFeatureType) -> MaterialFeaturizer:
        raise NotImplementedError


class M3GNetFeaturizer(MaterialFeaturizer):

    def featurize(self, structure: Structure) -> Any:
        raise NotImplementedError


class MaterialFeatureCollection(MaterialCollection):

    @property
    def features(self) -> MaterialFeatureType:
        raise NotImplementedError


class MlSplit(BaseModel):
    name: Split
    X: Any
    y: Any


class MlData(BaseModel):
    train: MlSplit
    val: MlSplit
    test: MlSplit


class XASData(BaseModel, MaterialFeatureCollection, AbsorptionSpectra):

    pass


class XASMlData(BaseModel, XASData, MlData):

    pass


class MaterialSplit:

    @staticmethod
    def split(XASData, split_fractions):
        raise NotImplementedError


class Outliers:

    @staticmethod
    def remove(XASData, std_factor: float = None):
        raise NotImplementedError

    @staticmethod
    def outliers(XASData, std_factor: float = None):
        raise NotImplementedError


class MLModel(ABC):

    @abstractmethod
    def fit(self, X: Any, y: Any):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    def load(self, path: Path) -> Self:
        raise NotImplementedError

    def save(self, path: Path) -> Self:
        raise NotImplementedError


class XASMlModel(XASMlData, MLModel):

    def fit(self, X: XASMlData.features, y: XASMlData.spectra) -> Self:
        raise NotImplementedError

    def predict(self, X: XASMlData.features) -> XASMlData.spectra:
        raise NotImplementedError


class XASBlockFactory:

    @staticmethod
    def block(config: XASBlockConfig) -> XASMlModel:
        raise NotImplementedError


class FileHander:
    raise NotImplementedError


class MLData(BaseModel):
    X: List[float]
    y: List[float]

    @field_validator("X", "y")
    @classmethod
    def _from_numpy(cls, v):
        if isinstance(v, np.ndarray):
            return [x.item() for x in v]
        return v


# %%
