# %%
from refactor.MlData import MLData
from typing import List
import numpy as np
from pydantic import Field, validate_call, BaseModel
from refactor.spectra_data import Spectrum


class OutlierDetector:

    @staticmethod
    def non_outliers(array: np.ndarray, std_factor: float) -> np.ndarray:
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        upper_bound = mean + std_factor * std
        lower_bound = mean - std_factor * std
        mask = np.all((array <= upper_bound) & (array >= lower_bound), axis=1)
        return mask

    @staticmethod
    def outliers(array: np.ndarray, std_factor: float) -> np.array:
        return ~OutlierDetector.non_outliers(array, std_factor)


# %%


if __name__ == "__main__":
    import unittest

    class TestOutlierRemover(unittest.TestCase):  # TODO: move to test file

        def test_outliers(self):
            spectra = np.random.rand(100, 5)
            spectra[0][0] += 10
            mask = OutlierDetector().outliers(spectra, 2)
            self.assertTrue(mask[0])
            self.assertFalse(np.any(mask[1:]))

    unittest.main(argv=[""], exit=False)

# %%

from refactor.spectra_enums import Element, SpectrumType
from refactor.spectra_data import ElementSpectrum
from refactor.io import FileHandler
from config.defaults import cfg
from p_tqdm import p_map
from refactor.spectra_enums import ElementsFEFF, ElementsVASP
import os


def remove_outliers_in_ml_data(element, spectra_type):
    load_file_handler = FileHandler(cfg.serialization)

    ml_file_paths = list(
        load_file_handler.serialized_objetcs_filepaths(
            MLData, element=element, type=spectra_type
        )
    )

    ml_data = [
        load_file_handler.deserialize_json(MLData, custom_filepath=fp)
        for fp in ml_file_paths
    ]

    std_factors = cfg.ml_data.anamoly_filter_std_cutoff
    std_factor = std_factors[spectra_type]

    spectras = np.array([data.y for data in ml_data])
    outliers = OutlierDetector().outliers(spectras, std_factor)

    files_to_remove = [fp for fp, o in zip(ml_file_paths, outliers) if o]
    print(f"Removing {len(files_to_remove)} outliers for {element} {spectra_type}")

    for fp in files_to_remove:
        os.remove(fp)

    files_after_removal = list(
        load_file_handler.serialized_objetcs_filepaths(
            MLData, element=element, type=spectra_type
        )
    )

    print(f"Files before removal: {len(ml_file_paths)}")
    print(f"Files after removal: {len(files_after_removal)}")
    print(f"Files removed: {len(ml_file_paths) - len(files_after_removal)}")
    print(f"Outliers: {sum(outliers)}")


# WARNING: DO NOT RUN TWICE
# for element in ElementsFEFF:
#     filter_outliers_and_save(element, SpectrumType.FEFF)
# for element in ElementsVASP:
#     filter_outliers_and_save(element, SpectrumType.VASP)

# %%
