# %%
import os

import numpy as np

from config.defaults import cfg
from refactor.io import FileHandler
from refactor.spectra_data import (
    ElementSpectrum,
    EnergyGrid,
    IntensityValues,
    Material,
    MaterialID,
    MaterialStructure,
)
from refactor.spectra_enums import Element, SpectrumType
from refactor.utilities._legacy.ml_data_generator import MLDataGenerator
from src.data.vasp_data_raw import RAWDataVASP


def save_spectra(spectra_type, file_handler):
    element_dict = {
        SpectrumType.FEFF: Element,
        SpectrumType.VASP: [Element.Cu, Element.Ti],
    }
    for element in element_dict[spectra_type]:

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
            ids_and_sites = [(a, b.split("_")[0]) for a, b in ids_and_sites]
            ids_and_sites = [(a, b.split("_")[0]) for a, b in ids_and_sites]
            poscar_paths = list(raw_data.poscar_paths.values())

        for id_and_site, poscar_path in zip(ids_and_sites, poscar_paths):
            id_string, site_string = id_and_site
            print("Processing:", id_string, site_string, element.name)

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
                    element_spectrum.__dict__.update(
                        {"index": site_index}
                    )  # bypass validation for saving
                file_handler.save(element_spectrum)
            except Exception as e:  # there are missing data/files
                print(f"Error processing {id_string} {site_string} {element.name}")
                print(e)


if __name__ == "__main__":

    save_spectra(
        spectra_type=SpectrumType.VASP,  # or SpectrumType.FEFF
        file_handler=FileHandler(config=cfg.serialization, replace_existing=False),
    )


# %%
