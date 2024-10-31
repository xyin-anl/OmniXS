# %%
import warnings
import re
from typing import List, Tuple

import numpy as np

from omnixas.data import (
    Element,
    MaterialSplitter,
    MLData,
    MLSplits,
    SpectrumType,
)
from omnixas.utils import DEFAULTFILEHANDLER, FileHandler
from omnixas.data import DataTag


class MLSplitGenerator:

    def __init__(self, file_handler: FileHandler = DEFAULTFILEHANDLER()):
        self.file_handler = file_handler

    def generate_ml_splits(
        self,
        data_tag: DataTag,
        target_fractions: List[float] = [0.8, 0.1, 0.1],
        seed: int = 42,
    ):
        element = data_tag.element
        type = data_tag.type
        file_paths = self.file_handler.serialized_objects_filepaths(
            MLData, element=element, type=type
        )
        idSites = [
            self._ml_filename_to_idSite(file_path, element, type)
            for file_path in file_paths
        ]
        train_idSites, val_idSites, test_idSites = MaterialSplitter.split(
            idSite=idSites,
            target_fractions=target_fractions,
            seed=seed,
        )
        train_data = self._to_ml_data(element, type, train_idSites)
        val_data = self._to_ml_data(element, type, val_idSites)
        test_data = self._to_ml_data(element, type, test_idSites)
        ml_splits = MLSplits(train=train_data, val=val_data, test=test_data)
        return ml_splits

    def _construct_filename(
        self,
        material_id: str,
        site: str,
        element: Element,
        spectra_type: SpectrumType,
    ) -> str:  # TODO: remove hardcoding
        warnings.warn("Hardcoded path in _construct_filename")
        return f"dataset/ml_data/{spectra_type.value}/{element.value}/{material_id}_site_{site}_{element.value}_{spectra_type.value}.json"

    def _idSite_to_ml_filename(
        self,
        idSite: Tuple[str, str],
        element: Element,
        spectra_type: SpectrumType,
    ) -> str:
        material_id, site = idSite
        return self._construct_filename(material_id, site, element, spectra_type)

    def _ml_filename_to_idSite(
        self,
        filename: str,
        element: Element,
        spectra_type: SpectrumType,
    ) -> Tuple[str, str]:
        # TODO: remove hardcoding
        pattern = f"dataset/ml_data/{spectra_type.value}/{element.value}/(.*)_site_(.+)_(.+)_(.+).json"
        match = re.search(pattern, filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        return match.group(1), match.group(2)

    def _load_ml_data(
        self,
        idSite: Tuple[str, str],
        element: Element,
        spectra_type: SpectrumType,
    ) -> MLData:
        filepath = self._idSite_to_ml_filename(idSite, element, spectra_type)
        return self.file_handler.deserialize_json(MLData, custom_filepath=filepath)

    def _to_ml_data(
        self,
        element: Element,
        spectra_type: SpectrumType,
        idSites: List[Tuple[str, str]],
    ) -> MLData:
        ml_data_list = [
            self._load_ml_data(idSite, element, spectra_type) for idSite in idSites
        ]
        X_train = np.stack([d.X for d in ml_data_list])
        y_train = np.stack([d.y for d in ml_data_list])
        return MLData(X=X_train, y=y_train)


def main():
    import tqdm
    from omnixas.data import AllDataTags

    for tag in tqdm.tqdm(AllDataTags()):
        split = MLSplitGenerator().generate_ml_splits(
            tag,
            target_fractions=[0.8, 0.1, 0.1],
            seed=42,
        )
        DEFAULTFILEHANDLER().serialize_json(
            split, supplemental_info={"element": tag.element, "type": tag.type}
        )


# %%

if __name__ == "__main__":
    # main()

    split = MLSplitGenerator().generate_ml_splits(
        DataTag(element=Element.Cu, type=SpectrumType.FEFF),
        target_fractions=[0.8, 0.1, 0.1],
        seed=42,
    )
    print(len(split.train.X), split.train.X[0])
