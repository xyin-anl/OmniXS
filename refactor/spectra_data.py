# %%
import json
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
    model_validator,
    validator,
)

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
    structure: Optional[Structure] = Field(None, description="Material structure")
    site: Optional[Site] = Field(None, description="Absorption site")

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


# %%

# @HumanReadable()
# class ElementSpectra(BaseModel, validate_assignment=True):
#     element: Element = Field(..., description="Element of the material")
#     type: SpectraType = Field(..., description="Type of Spectra")
#     spectra: Optional[List[Spectrum]] = Field(
#         None, description="Collection of spectra for the element"
#     )

#     def populate_spectra_from_materials(self, materials: List[Material]) -> Self:
#         self.spectra = [
#             m.site.spectra[self.type]
#             for m in materials
#             if self._material_has_required_spectrum(m, self.element, self.type)
#         ]
#         return self

#     @staticmethod
#     def _material_has_required_spectrum(
#         material: Material,
#         element: Element,
#         type: SpectraType,
#     ) -> bool:

#         if material.site is None:
#             raise ValueError(f"Absorption site not found in material {material.id}")
#         if material.site.element != element:
#             raise ValueError(f"Element {element} not found in material {material.id}")
#         if type not in material.site.spectra:
#             raise ValueError(f"{type} spectrum not found in material {material.id}")
#         return True


# %%

if __name__ == "__main__":

    materials = [
        Material(
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
    ]
    print(materials)

    # elementSpectra = ElementSpectra(
    #     element=Element.Ti,
    #     type=SpectraType.FEFF,
    # ).populate_spectra_from_materials(materials)
    # print(elementSpectra)
    # # serialization
    # with open(f"{elementSpectra.element}_{elementSpectra.type}.json", "w") as f:
    #     f.write(elementSpectra.json())
    # # deserialization
    # with open(f"{elementSpectra.element}_{elementSpectra.type}.json", "r") as f:
    #     data = f.read()
    #     elementSpectra_loaded = ElementSpectra(**json.loads(data))
    # print(elementSpectra_loaded)

# %%
