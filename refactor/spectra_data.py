# %%
from typing import Dict, List, Optional, Self, Union

import numpy as np
from box import Box  # converts dict keys to attributes (AttributeDict)
from matgl.ext.pymatgen import Structure
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
    validator,
    field_serializer,
)
from pymatgen.core import Structure as PymatgenStructure

from refactor.spectra_enums import Element, MaterialIDPrefix, SpectraType
from refactor.utils import HumanReadable


@HumanReadable()
class Spectrum(BaseModel, validate_assignment=True):
    type: SpectraType
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


@HumanReadable()
class Site(BaseModel):
    index: NonNegativeInt = Field(..., description="Absorption site index")
    element: Optional[Element] = Field(None, description="Element at absorption site")
    spectra: Optional[Union[Box, Dict[SpectraType, Spectrum]]] = Field(
        default_factory=Box, description="Spectra for the site"
    )

    @property
    def spectrum(self) -> Spectrum:
        if len(self.spectra) > 1:
            raise ValueError("Multiple spectra found for site")
        return list(self.spectra.values())[0]

    def assign_spectra(self, spectrum: Spectrum):
        self.spectra[spectrum.type] = spectrum

    class Config:
        arbitrary_types_allowed = True  # for Box


@HumanReadable()
class Material(BaseModel):
    id: str = Field(..., description="Material ID")
    structure: Optional[PymatgenStructure] = Field(
        None, description="Pymatgen structure"
    )
    site: Optional[Site] = Field(None, description="Absorption site")

    @field_serializer("structure")
    def serialize_structure(self, v):
        return v.as_dict()

    @field_validator("id")
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


if __name__ == "__main__":

    material = Material(
        id="mp-1234",
        site=Site(
            index=0,
            element=Element.Ti,
            spectra={
                SpectraType.FEFF: Spectrum(
                    type=SpectraType.FEFF,
                    energies=[1, 2, 3],
                    intensities=[4, 5, 6],
                )
            },
        ),
    )
    print(material)

# %%
