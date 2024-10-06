# %%
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
)
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Self, Union, Optional
from pathlib import Path

import numpy as np
from matgl.ext.pymatgen import Structure


# %%


class Element(str, Enum):
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"


class MaterialIDPrefix(str, Enum):
    mp = "mp"
    mvc = "mvc"  # depricated in materials project


class MaterialID(BaseModel):
    type: MaterialIDPrefix
    id: PositiveInt

    def __str__(self):
        return f"{self.type.name}-{self.id}"

    def __repr__(self):
        return str(self)


MaterialID(type="mp", id=1234)


# %%


class Simulation(str, Enum):
    VASP = "VASP"
    FEFF = "FEFF"


class Spectrum(BaseModel, validate_assignment=True):
    type: Simulation
    energies: List[NonNegativeFloat] = Field(None, min_length=1)
    intensities: List[NonNegativeFloat] = Field(None, min_length=1)

    @field_validator("energies")
    @classmethod
    def check_monotonic(cls, v):
        if not all(np.diff(v) > 0):
            raise ValueError("Energies must be monotonically increasing")
        return v

    @model_validator(mode="after")
    @classmethod
    def check_length(cls, v):
        if None in [v.energies, v.intensities]:
            return v
        if len(v.energies) != len(v.intensities):
            raise ValueError("Energies and intensities must have the same length")
        return v


# %%


class Site(BaseModel):
    site: Optional[NonNegativeInt] = Field(None, description="Site number")
    element: Optional[Element] = Field(None, description="Element at absorption site")
    spectra: Optional[List[Spectrum]] = Field(None, description="XAS spectra")

    def __str__(self) -> str:
        return f"site_{self.site:03d}_{self.element.name}"


s = Site()
s.element = Element.Ti
s.site = 22
s.spectra = [
    Spectrum(type=Simulation.FEFF, intensities=[1, 2, 3]),
    Spectrum(type=Simulation.VASP, intensities=[1, 2, 3], energies=[1, 2, 3]),
]
print(s)
s

# %%


class Material(BaseModel):
    id: MaterialID = Field(None, description="Material ID")
    structure: Optional[Structure] = Field(None, description="Material structure")
    site: Optional[Site] = Field(None, description="Absorption site")

    def __str__(self) -> str:
        return f"{self.id} {self.structure.formula}"


material = Material(id=MaterialID(type="mp", id=1234))
material

# %%


class ElementSpectra(BaseModel, validate_assignment=True):
    type: Optional[Simulation] = Field(None, description="Type of simulation")
    materials: Optional[List[Material]] = Field(
        [], description="Collection of materials", min_length=1
    )

    @field_validator("materials")
    def validate_same_element(cls, materials):
        if materials is not None:
            elements = set(
                material.site.element
                for material in materials
                if material.site is not None
            )
            if len(elements) > 1:
                raise ValueError("All materials must have the same element")
        return materials

    @model_validator(mode="after")
    def spectra_type_present_in_all(cls, values):
        if values.materials is not None:
            if any(
                material.site is None
                or material.site.spectra is None
                or any(
                    [spectrum.type != values.type for spectrum in material.site.spectra]
                )
                for material in values.materials
            ):
                raise ValueError(f"{values.type} spectra not found for all materials")

    @model_validator(mode="after")
    def spectra_count_matches_material_count(cls, values):
        if values is not None and values.materials is not None:
            if len(cls.spectra) != len(values.materials):
                raise ValueError("Spectra count does not match material count")

    @property
    def element(self) -> Optional[Element]:
        if self.materials:
            return self.materials[0].site.element
        return None

    @property
    def spectra(self):
        if self.materials is None:
            raise ValueError("Materials not loaded to access spectra")
        if self.type is None:
            raise ValueError("Spectra type not selected")
        spectra = [
            material.site.spectra
            for material in self.materials
            if material.site.spectra.type == self.type
        ]
        if len(spectra) != len(self.materials):  # double sanity check
            raise ValueError("Spectra not found for all materials")
        return spectra


ElementSpectra(
    type=Simulation.FEFF,
    materials=[
        Material(
            site=Site(element=Element.Ti, spectra=[Spectrum(type=Simulation.FEFF)])
        ),
        Material(
            site=Site(element=Element.Ti, spectra=[Spectrum(type=Simulation.FEFF)])
        ),
    ],
)


# # SHould raise error
# ElementSpectra(
#     type=Simulation.FEFF,
#     materials=[
#         Material(site=Site(spectra=[Spectrum(type=Simulation.FEFF)])),
#         Material(site=Site(spectra=[Spectrum(type=Simulation.VASP)])),
#     ],
# )


# %%


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


class XASBlockTypes(str, Enum):
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


class XASData(BaseModel, MaterialFeatureCollection, ElementSpectra):

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
