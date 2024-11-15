# %%
# from .enums import ElementsVASP, SpectrumType, ElementsFEFF
from enum import StrEnum

from ..data.ml_data import DataTag


class VASPDataTags:
    def __new__(cls):
        return [
            DataTag(element=element, type=SpectrumType.VASP) for element in ElementsVASP
        ]


class FEFFDataTags:
    def __new__(cls):
        return [
            DataTag(element=element, type=SpectrumType.FEFF) for element in ElementsFEFF
        ]


class AllDataTags:
    def __new__(cls):
        return FEFFDataTags() + VASPDataTags()


class ElementsFEFF(StrEnum):
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"


class ElementsVASP(StrEnum):
    Ti = "Ti"
    Cu = "Cu"


class SpectrumType(StrEnum):
    VASP = "VASP"
    FEFF = "FEFF"


class Element(StrEnum):
    Ti = "Ti"
    V = "V"
    Cr = "Cr"
    Mn = "Mn"
    Fe = "Fe"
    Co = "Co"
    Ni = "Ni"
    Cu = "Cu"
    All = "All"  # placeholder for universalXAS


Ti = Element.Ti
V = Element.V
Cr = Element.Cr
Mn = Element.Mn
Fe = Element.Fe
Co = Element.Co
Ni = Element.Ni
Cu = Element.Cu

FEFF = SpectrumType.FEFF
VASP = SpectrumType.VASP

# %%
