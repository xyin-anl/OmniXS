import os
import shutil
import warnings


def fix_m3gnet_version(
    model_name="M3GNet-MP-2021.2.8-PES",
    cache_dir=None,
):
    """temporary fix to use old version. (Bug reported)[t.ly/jIZdT]"""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matgl", model_name)

    valid_model_dir = os.path.join("scripts", model_name)
    if not os.path.exists(valid_model_dir):
        raise FileNotFoundError(f"Model directory {valid_model_dir} does not exist. ")

    try:
        shutil.copytree(valid_model_dir, cache_dir)
    except FileExistsError:
        warnings.warn(f"Model directory {cache_dir} already exists. ")
        warnings.warn("Overwriting existing model. ")
        shutil.rmtree(cache_dir)
        shutil.copytree(valid_model_dir, cache_dir)
    except Exception as e:
        raise e


if __name__ == "__main__":
    from DigitalBeamline.digitalbeamline.extern.m3gnet.featurizer import (
        featurize_material,
    )
    from pymatgen.core.structure import Structure

    fix_m3gnet_version()  # <--- this is the fix
    poscar_path = "dataset/VASP-raw-data/Cu/mp-1478/VASP/004_Cu/POSCAR"
    structure = Structure.from_file(poscar_path)
    features = featurize_material(structure)
    print(features.shape)
