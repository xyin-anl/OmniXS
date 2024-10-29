# %%
from typing import Dict, List, Optional, Self, Union, Literal
from pathlib import Path

import numpy as np
from box import Box  # converts dict keys to attributes (AttributeDict)
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    NonNegativeFloat,
    NonNegativeInt,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
    validator,
    field_serializer,
    RootModel,
    ValidationInfo,
)
from pymatgen.core import Structure as PymatgenStructure

from omnixas.data import Element, MaterialIDPrefix, SpectrumType
from omnixas.utils.readable_enums import ReadableEnums


class MaterialID(RootModel, validate_assignment=True):
    root: str = Field(..., description="Material ID")

    @field_validator("root")
    @classmethod
    def check_id(cls, v):
        if not v:
            raise ValueError("Material ID cannot be empty")
        try:
            prefix, id_str = v.split("-")
            MaterialIDPrefix(prefix)
            int(id_str)
        except ValueError:
            msg = "Material ID must be in the format: <prefix>-<id>"
            msg += f" <prefix> must be one of {[m.value for m in MaterialIDPrefix]}"
            msg += " <id> must be a positive integer"
            raise ValueError(msg)
        return v

    def __str__(self):
        return self.root

    def __repr__(self):
        return self.root

    def __hash__(self):
        return hash(self.root)

    def __eq__(self, other):
        return self.root == other.root


@ReadableEnums()
class EnergyGrid(RootModel, validate_assignment=True):
    root: List[NonNegativeFloat] = Field(None, min_length=1)

    @field_validator("root")
    @classmethod
    def check_monotonic(cls, v):
        if not all(np.diff(v) > 0):
            raise ValueError("Energies must be monotonically increasing")
        return v

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)


@ReadableEnums()
class IntensityValues(RootModel, validate_assignment=True):
    root: List[NonNegativeFloat] = Field(None, min_length=1)

    @field_validator("root")
    @classmethod
    def check_non_negative(cls, v):
        if not all(i >= 0 for i in v):
            raise ValueError("All intensities must be non-negative")
        return v

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)


@ReadableEnums()
class Spectrum(BaseModel, validate_assignment=True):
    intensities: IntensityValues
    energies: Optional[EnergyGrid]

    @model_validator(mode="after")
    def check_length(cls, values):
        energies = values.energies
        intensities = values.intensities
        if energies is not None and len(energies) != len(intensities):
            raise ValueError("Energies and intensities must have the same length")
        return values


@ReadableEnums()
class MaterialStructure(RootModel, validate_assignment=True):
    root: PymatgenStructure = Field(None, description="Pymatgen structure")

    @classmethod
    def from_file(cls, path: Path):
        return cls(root=PymatgenStructure.from_file(path))

    @field_serializer("root")
    def serialize_structure(self, pymatgen_structure):
        return pymatgen_structure.as_dict()

    @property
    def sites(self):
        return self.root.sites

    def __eq__(self, other):
        return self.root.as_dict() == other.root.as_dict()


@ReadableEnums()
class Material(BaseModel):
    id: MaterialID
    structure: Optional[MaterialStructure] = Field(None)

    def __name__(self):
        return self.id.root


from pymatgen.core.sites import PeriodicSite


@ReadableEnums()
class PymatgenSite(RootModel, validate_assignment=True):
    root: PeriodicSite = Field(None, description="Pymatgen site")

    @model_serializer()
    def serialize_model(self):
        return self.as_dict()

    @classmethod
    def from_site_index(cls, structure: MaterialStructure, index: int):
        return cls(structure.sites[index])


@ReadableEnums()
class SiteSpectrum(Spectrum):
    type: SpectrumType
    index: NonNegativeInt
    material: Optional[Material]

    @property
    def site(self) -> PymatgenSite:
        return self.material.structure.sites[self.index]

    @property
    def index_string(self) -> str:  # helper for file i/o
        return f"{self.index:03d}"


@ReadableEnums()
class ElementSpectrum(SiteSpectrum, validate_assignment=True):
    element: Element

    @model_validator(mode="after")
    def validate_element(self) -> Self:
        asked_element = self.element
        if self.type != SpectrumType.VASP:  # TODO: remove in deployment
            # coz vasp sims were done this way
            site_element = Element(ElementSpectrum.extract_element(self.site))
            site_element = Element(site_element)
            if asked_element != site_element:
                raise ValueError(
                    f"Element {asked_element} does not match site element {site_element}"
                )
        return self

    @staticmethod
    def extract_element(site: PymatgenSite) -> str:
        species = site.species
        if not species.is_element:
            raise ValueError("Site must be an element")
        return str(list(dict(species).keys())[0])


# loaded_spectrum = file_handler.load(
#     model=ElementSpectrum,
#     element=element,
#     material=Material(id=material_id),
#     index=site_index,
#     type=spectra_type,
# )
# assert element_spectrum == loaded_spectrum

# %%

# print([x for x in dir(PymatgenStructure) if "__" not in x])

# %%
