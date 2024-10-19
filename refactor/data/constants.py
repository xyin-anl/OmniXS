# %%
from refactor.data import Element, ElementsVASP, SpectrumType
from refactor.data.ml_data import DataTag

ElementsFEFF = Element


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

FEFFDataTags = [
    DataTag(element=element, type=SpectrumType.FEFF) for element in ElementsFEFF
]
VASPDataTags = [
    DataTag(element=element, type=SpectrumType.VASP) for element in ElementsVASP
]



