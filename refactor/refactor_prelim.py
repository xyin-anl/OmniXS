from pydantic import BaseModel, PositiveInt, Field, validator
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Self, Union
from pathlib import Path

import numpy as np
from matgl.ext.pymatgen import Structure


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    NONE = "none"


class DScribeFeaturizerType(str, Enum):
    SOAP = "SOAP"
    ACSF = "ACSF"
    LMBTR = "LMBTR"


class MaterialFeatureType(str, Enum):
    M3GNET = "M3GNet"
    DSCRIBE = "DScribe"


class MaterialIDPrefix(str, Enum):
    mp = "mp"
    mvc = "mvc"  # depricated in materials project


class XASBlockTypes(str, Enum):
    EXPERT = "expert"
    UNIVERSAL = "universal"
    TUNED = "tuned"


class Element(str, Enum):
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"


class Simulation(str, Enum):
    VASP = "VASP"
    FEFF = "FEFF"


class SpectraType(str, Enum):
    SIMULATION = "simulation"


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
    elements: List[Element] = Field(..., min_length=1, unique_items=True)
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


class Spectrum:

    @property
    def type(self) -> SpectraType:
        raise NotImplementedError

    @property
    def energies(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def intensities(self) -> np.ndarray:
        raise NotImplementedError


class Site:

    @property
    def site(self) -> int:
        raise NotImplementedError

    @property
    def element(self) -> Element:
        raise NotImplementedError

    @property
    def spectra(self) -> List[Spectrum]:
        raise NotImplementedError


class MaterialID(BaseModel):
    name: MaterialIDPrefix
    id: PositiveInt


class Material:

    @property
    def id(self) -> MaterialID:
        raise NotImplementedError

    @property
    def structure(self) -> Structure:
        raise NotImplementedError

    @property
    def sites(self) -> List[Site]:
        raise NotImplementedError


class MaterialCollection:

    @property
    def materials(self) -> List[Material]:
        raise NotImplementedError


class SiteSpectraCollection(MaterialCollection):

    @property
    def element(self) -> Element:
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

    @property
    def spectra(self):
        raise NotImplementedError


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


class XASData(BaseModel, MaterialFeatureCollection, SiteSpectraCollection):

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


# %%
