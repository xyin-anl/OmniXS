# %%
from loguru import logger
import os
from typing import List, Optional

import numpy as np
from p_tqdm import p_map

from config.defaults import cfg
from omnixas.data import (
    ElementSpectrum,
    EnergyGrid,
    IntensityValues,
    Material,
    MaterialID,
    MaterialStructure,
)
from omnixas.utils.constants import SpectrumType
from omnixas.utils.constants import Element
from omnixas.utils import DEFAULTFILEHANDLER
from omnixas._legacy.scripts.ml_data_generator import (
    MLDataGenerator,
)  # TODO: Depricate this, use fn in MLSplitGenerator
from _legacy.data.vasp_data_raw import RAWDataVASP

# %%


def save_element_spectra(spectra_type, element):
    if spectra_type == SpectrumType.FEFF:
        data_dir = os.path.dirname(cfg.paths.processed_data).format(
            compound=element, simulation_type=spectra_type
        )
        ids_and_sites = MLDataGenerator.parse_ids_and_site(element, data_dir)
        poscar_paths = [
            cfg.paths.poscar.format(compound=element, id=id_string)
            for id_string, _ in ids_and_sites
        ]
        poscar_paths = [path for path in poscar_paths if os.path.exists(path)]

    elif spectra_type == SpectrumType.VASP:
        raw_data = RAWDataVASP(compound=element, simulation_type=spectra_type)
        ids_and_sites = list(raw_data.poscar_paths.keys())
        ids_and_sites = [(i, site.split("_")[0]) for i, site in ids_and_sites]
        poscar_paths = list(raw_data.poscar_paths.values())

    out = p_map(
        save_spectrum,
        [spectra_type] * len(ids_and_sites),
        [element] * len(ids_and_sites),
        ids_and_sites,
        poscar_paths,
    )
    return out


def save_spectrum(spectra_type, element, id_and_site, poscar_path):
    id_string, site_string = id_and_site
    element = Element(element)

    output = []  # for debugging
    try:
        material_strucutre = MaterialStructure.from_file(poscar_path)
        material_id = MaterialID(id_string)
        material = Material(id=material_id, structure=material_strucutre)

        spectra_path = cfg.paths.processed_data.format(
            compound=element,
            simulation_type=spectra_type,
            id=id_string,
            site=site_string,
        )
        spectra_data = np.loadtxt(spectra_path)

        site_index = int(site_string)

        element_spectrum = ElementSpectrum(
            element=element,
            type=spectra_type,
            index=(
                site_index if spectra_type == SpectrumType.FEFF else 0
            ),  # coz sim was done this way
            material=material,
            intensities=IntensityValues(spectra_data[:, 1]),
            energies=EnergyGrid(spectra_data[:, 0]),
        )

        if spectra_type == SpectrumType.VASP:
            # bypass validation for saving because of the way the data was saved
            element_spectrum.__dict__.update(
                {"index": site_index}
            )  # TODO: remove in deployment

        save_path = (
            None
            if spectra_type == SpectrumType.FEFF
            else f"dataset/spectra/VASP/{element}/{id_string}_site_{site_string}_{element}_VASP.json"
        )  # hardcoded because of the way the data was saved TODO: fix this

        output.append(material)

        DEFAULTFILEHANDLER.serialize_json(element_spectrum, custom_filepath=save_path)

    except Exception as e:  # there are missing data/files. Ignore them.
        logger.warning(f"Error: {e}")
    return output


def save_spectra(
    spectra_type,
    elements: Optional[List[Element]] = None,
):
    for element in elements:
        out = save_element_spectra(spectra_type, element)
    return out


save_spectra(spectra_type=SpectrumType.VASP, elements=[Element.Cu])
# save_spectra(spectra_type=SpectrumType.FEFF, elements=[Element.Cu])

# %%
