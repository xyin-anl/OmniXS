import re
from p_tqdm import p_map
from tqdm import tqdm
from config.defaults import cfg
import os
import numpy as np
from DigitalBeamline.digitalbeamline.extern.m3gnet.featurizer import featurize_material
from pymatgen.core.structure import Structure


class MLDataGenerator:
    """Contains methods to prepare data for ML"""

    @staticmethod
    def save(compound, simulation_type, n_blocks=None):
        # avoid overwriting
        save_file = cfg.paths.ml_data.format(
            compound=compound, simulation_type=simulation_type
        )
        if os.path.exists(save_file):
            raise FileExistsError(f"File already exists: {save_file}")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        ml_data = MLDataGenerator.prepare(compound, simulation_type, n_blocks)
        ids, sites, features, energies, spectras = map(np.array, zip(*ml_data))
        for energy in energies:
            if not np.all(energy == energies[0]):
                raise ValueError("Not all energies are same")

        np.savez_compressed(
            save_file,
            ids=ids,
            sites=sites,
            energies=energies[0],  # coz energy grid verified to be same
            features=features,
            spectras=spectras,
        )

    @staticmethod
    def prepare(compound, simulation_type, n_blocks=None):
        data_dir = os.path.dirname(cfg.paths.processed_data).format(
            compound=compound, simulation_type=simulation_type
        )
        ids_and_sites = MLDataGenerator.parse_ids_and_site(compound, data_dir)
        ml_data = p_map(
            lambda x: (
                x[0],
                x[1],
                MLDataGenerator.featurize(compound, x[0], x[1], n_blocks),
                *MLDataGenerator.load_processed_data(
                    compound, x[0], x[1], simulation_type
                ).T,
            ),
            ids_and_sites[:],
        )
        return ml_data

    @staticmethod
    def featurize(compound: str, id: str, site_str: str, n_blocks=None):
        structure = MLDataGenerator.get_structure(compound, id)  # reads from POSCAR
        features_all = featurize_material(structure, n_blocks=n_blocks)
        # check if site infor from folder is valid for structure derived from POSCAR
        site = int(site_str)
        site_is_in_strucutre = site >= 0 and site < features_all.shape[0]
        site_is_of_correct_element = structure.species[site].symbol == compound
        if not site_is_in_strucutre or not site_is_of_correct_element:
            raise ValueError(f"Site {site} is not valid for {id}")
        return features_all[site]

    @staticmethod
    def get_structure(compound: str, id: str):
        poscar_path = cfg.paths.poscar.format(compound=compound, id=id)
        if not os.path.exists(poscar_path):
            raise FileNotFoundError(f"POSCAR not found for {id}")
        return Structure.from_file(poscar_path)

    @staticmethod
    def parse_ids_and_site(compound, dir_path):
        return [
            MLDataGenerator.filename_to_uid(compound, x)
            for x in os.listdir(dir_path)
            if x.endswith(".dat")
        ]

    @staticmethod
    def uid_to_filename(compound, id: str, site: str):
        if len(site) != 3:
            raise ValueError("Site info is assumed to be of form 'xxx'")
        return f"{id}_site_{site}_{compound}.dat"

    @staticmethod
    def filename_to_uid(compound, filename):
        id, site = re.split(rf"_site_|_{compound}\.dat", filename)[:2]
        return id, site

    @staticmethod
    def load_processed_data(compound, id, site, simulation_type):
        file_path = cfg.paths.processed_data.format(
            compound=compound,
            simulation_type=simulation_type,
            id=id,
            site=site,
        )
        return np.loadtxt(file_path)


if __name__ == "__main__":
    simulation_type = "FEFF"

    # compounds = ["Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Ti", "V"]
    # for compound in compounds:
    #     print(f"Preparing data for {compound}")
    #     MLDataGenerator.save(compound=compound, simulation_type=simulation_type)

    data3 = MLDataGenerator.prepare("Co", simulation_type, n_blocks=3)
    data2 = MLDataGenerator.prepare("Co", simulation_type, n_blocks=2)
    print("dummy")
