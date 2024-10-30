# %%
from .enums import Element, ElementsVASP, SpectrumType, ElementsFEFF
from .ml_data import DataTag


FEFF = SpectrumType.FEFF
VASP = SpectrumType.VASP

Ti = Element.Ti
V = Element.V
Cr = Element.Cr
Mn = Element.Mn
Fe = Element.Fe
Co = Element.Co
Ni = Element.Ni
Cu = Element.Cu


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
