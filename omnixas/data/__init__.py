# TODO: remove from release
from ..utils.constants import Element
from ..utils.constants import SpectrumType
from ..utils.constants import (
    FEFF,
    VASP,
    ElementsFEFF,
    ElementsVASP,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
    FEFFDataTags,
    VASPDataTags,
    AllDataTags,
)

from .ml_data import MLData, DataTag, MLSplits
from .xas import (
    MaterialID,
    MaterialStructure,
    Material,
    PymatgenSite,
    SiteSpectrum,
    ElementSpectrum,
    Spectrum,
    EnergyGrid,
    IntensityValues,
)

from .scaler import (
    ScaledMlSplit,
    IdentityScaler,
    UncenteredRobustScaler,
    ThousandScaler,
    MultiplicativeScaler,
)

from .material_split import MaterialSplitter

# from .merge_ml_splits import (
#     FEFFSplits,
#     MergedSplits,
#     VASPSplits,
# )  # Slows down the import


__all__ = [
    "Element",
    "ElementsVASP",
    "SpectrumType",
    "ElementsFEFF",
    "FEFF",
    "VASP",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "FEFFDataTags",
    "VASPDataTags",
    "MLData",
    "DataTag",
    "MLSplits",
    "MaterialID",
    "MaterialStructure",
    "Material",
    "PymatgenSite",
    "SiteSpectrum",
    "ElementSpectrum",
    "Spectrum",
    "EnergyGrid",
    "IntensityValues",
    # "MergedSplits",
    # "FEFFSplits",
    # "VASPSplits",
    "ScaledMlSplit",
    "IdentityScaler",
    "UncenteredRobustScaler",
    "ThousandScaler",
    "MultiplicativeScaler",
    "AllDataTags",
    "MaterialSplitter",
]
