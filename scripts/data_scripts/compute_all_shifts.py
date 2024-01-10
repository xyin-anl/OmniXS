from utils.src.misc import icecream
import multiprocessing
from src.data.feff_data_raw import RAWDataFEFF
from src.data.feff_data import FEFFData
from src.data.vasp_data_raw import RAWDataVASP
from src.data.vasp_data import VASPData
import numpy as np
from tqdm import tqdm
import os
import re


def ids_from_files(compound, simulation_type):
    processed_data_dir = os.path.join(
        "dataset",
        f"{simulation_type}-processed-data",
        compound,
    )
    files = os.listdir(processed_data_dir)
    reg_exp = re.compile(r"mp-.*_site_.*\.dat")
    files = list(filter(reg_exp.match, files))
    ids = list(map(lambda x: x.split(".")[0], files))
    return ids


def common_ids(compound):
    feff_ids = ids_from_files(compound=compound, simulation_type="FEFF")
    vasp_ids = ids_from_files(compound=compound, simulation_type="VASP")
    common_ids = set(feff_ids).intersection(set(vasp_ids))
    common_ids = list(map(lambda x: tuple(x.split("_site_")), common_ids))
    return common_ids


def compute_shift(params):
    compound, id = params
    # id = tuple(id.split("_site_"))
    data_vasp = VASPData(compound).load(id)
    data_feff = FEFFData(compound).load(id)
    alignment_shift = data_feff.align(data_vasp)
    return id, alignment_shift


def compute_all_shifts(compound, max_length=None, paralleize=True):
    common_ids = common_ids(compound)
    common_ids = list(common_ids)[:max_length]  # TODO: remove this line
    params = [(compound, id) for id in common_ids]
    if paralleize:
        with multiprocessing.Pool(processes=10) as pool:
            # shifts = pool.map(compute_shift, params)
            shifts = list(
                tqdm(
                    pool.imap_unordered(compute_shift, params),
                    total=len(params),
                )
            )
    else:
        shifts = map(compute_shift, params)
    shifts = dict(shifts)
    return shifts


def save_shifts(compound, shifts):
    site_ids = []
    site_shifts = []
    for id, shift in shifts.items():
        site_ids.append("_site_".join(id))
        site_shifts.append(shift)
    file_name = f"shifts_{compound}.txt"
    np.savetxt(
        file_name,
        np.vstack((site_ids, site_shifts)).T,
        fmt="%s",
        delimiter="\t",
        header="id_site\talignment_shift",
    )


if __name__ == "__main__":
    compound = "Ti"
    shifts = compute_all_shifts(compound, max_length=None)
    save_shifts(compound, shifts)

    compound = "Cu"
    shifts = compute_all_shifts(compound, max_length=None)
    save_shifts(compound, shifts)
