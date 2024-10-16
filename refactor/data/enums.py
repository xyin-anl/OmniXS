from enum import StrEnum


class MaterialIDPrefix(StrEnum):
    mp = "mp"
    mvc = "mvc"  # depricated in materials project


class Element(StrEnum):
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
