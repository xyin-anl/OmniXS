# %%
import os
from loguru import logger

import numpy as np

from config.defaults import cfg
from omnixas.core.constants import SpectrumType
from omnixas.core.constants import ElementsFEFF, ElementsVASP
from omnixas.data.xas import ElementSpectrum
from omnixas.core.constants import Element
from omnixas.data.ml_data import MLData
from omnixas.utils import DEFAULTFILEHANDLER, FileHandler
from omnixas.utils.spectra_outliers import OutlierDetector


def remove_outliers_in_ml_data(element, spectra_type):

    ml_file_paths = DEFAULTFILEHANDLER.serialized_objects_filepaths(
        MLData, element=element, type=spectra_type
    )

    ml_data = [
        DEFAULTFILEHANDLER.deserialize_json(MLData, custom_filepath=fp)
        for fp in ml_file_paths
    ]

    std_factors = cfg.ml_data.anamoly_filter_std_cutoff  # TODO: replace with CONFIGS
    std_factor = std_factors[spectra_type]

    spectras = np.array([data.y for data in ml_data])
    outliers = OutlierDetector().outliers(spectras, std_factor)

    files_to_remove = [fp for fp, o in zip(ml_file_paths, outliers) if o]
    logger.info(f"{len(files_to_remove)} outliers for {element} {spectra_type}")
    logger.info("Uncomment to remove")

    # print(f"Removing {len(files_to_remove)} outliers for {element} {spectra_type}")
    # for fp in files_to_remove:
    #     os.remove(fp)
    # files_after_removal = list(
    #     load_file_handler.serialized_objetcs_filepaths(
    #         MLData, element=element, type=spectra_type
    #     )
    # )
    # print(f"Files before removal: {len(ml_file_paths)}")
    # print(f"Files after removal: {len(files_after_removal)}")
    # print(f"Files removed: {len(ml_file_paths) - len(files_after_removal)}")
    # print(f"Outliers: {sum(outliers)}")


for element in ElementsFEFF:
    remove_outliers_in_ml_data(element, SpectrumType.FEFF)

for element in ElementsVASP:
    remove_outliers_in_ml_data(element, SpectrumType.VASP)


# %%


def remove_outliers_in_spectrum(element, spectra_type):
    load_file_handler = DEFAULTFILEHANDLER

    spectrum_file_paths = load_file_handler.serialized_objects_filepaths(
        ElementSpectrum, element=element, type=spectra_type
    )

    spectra = [
        load_file_handler.deserialize_json(ElementSpectrum, custom_filepath=fp)
        for fp in spectrum_file_paths
    ]

    std_factors = cfg.ml_data.anamoly_filter_std_cutoff
    std_factor = std_factors[spectra_type]

    spectras = np.array([np.array(data.intensities) for data in spectra])
    outliers = OutlierDetector().outliers(spectras, std_factor)

    files_to_remove = [fp for fp, o in zip(spectrum_file_paths, outliers) if o]
    print(f"{len(files_to_remove)} outliers for {element} {spectra_type}")
    print("Uncomment to remove")

    # print(f"Removing {len(files_to_remove)} outliers for {element} {spectra_type}")
    # for fp in files_to_remove:
    #     os.remove(fp)
    # files_after_removal = list(
    #     load_file_handler.serialized_objetcs_filepaths(
    #         ElementSpectrum, element=element, type=spectra_type
    #     )
    # )
    # print(f"Files before removal: {len(spectrum_file_paths)}")
    # print(f"Files after removal: {len(files_after_removal)}")
    # print(f"Files removed: {len(spectrum_file_paths) - len(files_after_removal)}")
    # print(f"Outliers: {sum(outliers)}")


for element in ElementsFEFF:
    remove_outliers_in_spectrum(element, SpectrumType.FEFF)
for element in ElementsVASP:
    remove_outliers_in_spectrum(element, SpectrumType.VASP)
