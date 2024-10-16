# %%
import numpy as np
import re
from typing import List, Tuple

from refactor.data.constants import ElementsFEFF
from refactor.data.ml_data import MLSplits
from refactor.data.ml_data import MLData
from src.data.material_split import MaterialSplitter
from refactor.utilities.io import FileHandler, DEFAULTFILEHANDLER
from refactor.data.enums import Element, SpectrumType


def construct_filename(
    material_id: str, site: str, element: Element, spectra_type: SpectrumType
) -> str:  # TODO: remove hardcoding
    return f"dataset/ml_data/{spectra_type.value}/{element.value}/{material_id}_site_{site}_{element.value}_{spectra_type.value}.json"


def idSite_to_ml_filename(
    idSite: Tuple[str, str], element: Element, spectra_type: SpectrumType
) -> str:
    material_id, site = idSite
    return construct_filename(material_id, site, element, spectra_type)


def ml_filename_to_idSite(
    filename: str, element: Element, spectra_type: SpectrumType
) -> Tuple[str, str]:
    # TODO: remove hardcoding
    pattern = f"dataset/ml_data/{spectra_type.value}/{element.value}/(.*)_site_(.+)_(.+)_(.+).json"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return match.group(1), match.group(2)


def load_ml_data(
    idSite: Tuple[str, str], element: Element, spectra_type: SpectrumType
) -> MLData:
    filepath = idSite_to_ml_filename(idSite, element, spectra_type)
    return DEFAULTFILEHANDLER.deserialize_json(MLData, custom_filepath=filepath)


def to_ml_data(
    element: Element, spectra_type: SpectrumType, idSites: List[Tuple[str, str]]
) -> MLData:
    ml_data_list = [load_ml_data(idSite, element, spectra_type) for idSite in idSites]
    X_train = np.stack([d.X for d in ml_data_list])
    y_train = np.stack([d.y for d in ml_data_list])
    return MLData(X=X_train, y=y_train)


def main(element: Element, spectra_type: SpectrumType):

    file_paths = DEFAULTFILEHANDLER.serialized_objects_filepaths(
        MLData, element=element, type=spectra_type
    )

    idSites = [
        ml_filename_to_idSite(file_path, element, spectra_type)
        for file_path in file_paths
    ]

    train_idSites, val_idSites, test_idSites = MaterialSplitter.split(
        idSite=idSites, target_fractions=[0.8, 0.1, 0.1]
    )

    train_data = to_ml_data(element, spectra_type, train_idSites)
    val_data = to_ml_data(element, spectra_type, val_idSites)
    test_data = to_ml_data(element, spectra_type, test_idSites)

    ml_splits = MLSplits(train=train_data, val=val_data, test=test_data)

    DEFAULTFILEHANDLER.serialize_json(
        ml_splits, {"element": element, "type": spectra_type}
    )


# %%

if __name__ == "__main__":
    from refactor.data.enums import ElementsVASP, SpectrumType
    from tqdm import tqdm

    for element in tqdm(ElementsFEFF, "FEFF"):
        main(element, SpectrumType.FEFF)

    for element in tqdm(ElementsVASP, "VASP"):
        main(element, SpectrumType.VASP)

    ml_splits = DEFAULTFILEHANDLER.deserialize_json(
        MLSplits, supplemental_info={"element": Element.Cu, "type": SpectrumType.VASP}
    )

# %%
