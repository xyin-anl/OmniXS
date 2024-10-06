# %%
import json
from box import Box  # converts dict keys to attributes (AttributeDict)
from typing import Dict
from pydantic import (
    BaseModel,
    Field,
    validator,
    NonNegativeFloat,
    NonNegativeInt,
    field_validator,
    model_validator,
    computed_field,
)
from typing import Any, List, Self, Union, Optional
from pathlib import Path
from refactor.spectra_enums import (
    Element,
    MaterialIDPrefix,
    SpectraType,
    FEFF,
    VASP,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
)

import numpy as np
from matgl.ext.pymatgen import Structure

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


@HumanReadable()
class ElementSpectra(BaseModel, validate_assignment=True):
    element: Element = Field(..., description="Element of the material")
    type: SpectraType = Field(..., description="Type of Spectra")
    spectra: Optional[List[Spectrum]] = Field(
        None, description="Collection of spectra for the element"
    )
    materials: Optional[List[Material]] = Field(
        None, description="Collection of materials"
    )

    @field_validator("materials")
    @classmethod
    def _validate_materials(cls, materials, values):
        if materials is None:
            return materials
        element = cls._get_common_element(materials)
        if element != values.data["element"]:
            raise ValueError("Element does not match the materials")
        # update spectra
        values.data["spectra"] = cls._extract_spectra(materials, values.data["type"])
        return materials

    @staticmethod
    def _get_common_element(materials: List[Material]) -> Element:
        elements = set(
            material.site.element for material in materials if material.site is not None
        )
        if len(elements) > 1:
            raise ValueError("All materials must have the same element")
        return next(iter(elements))

    @classmethod
    def _extract_spectra(
        cls, materials: List[Material], spectra_type: SpectraType
    ) -> List[Spectrum]:
        spectra = []
        for material in materials:
            spectrum = cls._get_spectrum_for_material(material, spectra_type)
            # spectrum_dict = spectrum.dict()
            # del spectrum_dict["type"]  # Exclude the type property
            spectra.append(spectrum)
        return spectra

    @classmethod
    def _get_spectrum_for_material(cls, material: Material, spectra_type: SpectraType):
        if material.site is None or material.site.spectra is None:
            raise ValueError(f"{cls.type} spectra not found for material")

        spectrum = next(
            (s for s in material.site.spectra.values() if s.type == spectra_type), None
        )
        if spectrum is None:
            raise ValueError(f"{cls.type} spectra not found for material")

        return spectrum


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

    elementSpectra = ElementSpectra(
        element=Element.Ti,
        type=SpectraType.FEFF,
        materials=materials,
    )
    print(elementSpectra)

    # serialization
    with open(f"{elementSpectra.element}_{elementSpectra.type}.json", "w") as f:
        f.write(elementSpectra.json())

    # deserialization

    with open(f"{elementSpectra.element}_{elementSpectra.type}.json", "r") as f:
        data = f.read()
        elementSpectra_loaded = ElementSpectra(**json.loads(data))
    print(elementSpectra_loaded)

    # %%
