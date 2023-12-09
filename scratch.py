# %%
from utils.src.plots.highlight_tick import highlight_tick
import multiprocessing
import re
from tqdm import tqdm
import os
import time
import pickle
import importlib
import src
from utils.src.misc import icecream
from utils.src.plots.highlight_tick import highlight_tick
from matplotlib import pyplot as plt
from src.data.feff_data_raw import RAWDataFEFF
import random
from copy import deepcopy
import scienceplots
import matplotlib as mpl
import numpy as np
from src.data.feff_data import FEFFData
from src.data.vasp_data_raw import RAWDataVASP
from itertools import combinations_with_replacement
from pprint import pprint
from typing import Tuple, TypedDict, Union
import lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
import torch
import yaml
from scipy.signal import convolve
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scripts.pca_plots import linear_fit_of_pcas, plot_pcas
from scripts.plots_model_report import (
    heatmap_of_lines,
    plot_predictions,
    # plot_residue_histogram,
    plot_residue_cv,
    plot_residue_heatmap,
    plot_residue_quartiles,
)
from src.ckpt_predictions import get_optimal_fc_predictions
from src.plot.model_report import linear_model_predictions, model_report
from src.data.vasp_data import VASPData
from src.pl_data import XASData
from utils.src.lightning.pl_module import PLModule
from utils.src.optuna.dynamic_fc import PlDynamicFC
from utils.src.plots.highlight_tick import highlight_tick

imports_1 = [
    "src",
    "utils.src.misc",
    "utils.src.plots.highlight_tick",
    "matplotlib.pyplot",
    "src.data.feff_data_raw",
    "scienceplots",
    "matplotlib",
    "src.data.feff_data",
    "src.data.vasp_data_raw",
    "scripts.pca_plots",
    "scripts.plots_model_report",
    "src.ckpt_predictions",
    "src.plot.model_report",
    "src.data.vasp_data",
    "src.pl_data",
    "utils.src.lightning.pl_module",
    "utils.src.optuna.dynamic_fc",
]


imports_2 = {
    "utils.src.misc": ["icecream"],
    "utils.src.plots.highlight_tick": ["highlight_tick"],
    "src.data.feff_data_raw": ["RAWDataFEFF"],
    "src.data.feff_data": ["FEFFData"],
    "src.data.vasp_data_raw": ["RAWDataVASP"],
    "src.data.vasp_data": ["VASPData"],
    "src.pl_data": ["XASData"],
    "utils.src.lightning.pl_module": ["PLModule"],
    "utils.src.optuna.dynamic_fc": ["PlDynamicFC"],
    "scripts.pca_plots": ["linear_fit_of_pcas", "plot_pcas"],
    "scripts.plots_model_report": [
        "heatmap_of_lines",
        "plot_predictions",
        "plot_residue_cv",
        "plot_residue_heatmap",
        "plot_residue_quartiles",
    ],
    "src.ckpt_predictions": ["get_optimal_fc_predictions"],
    "src.plot.model_report": ["linear_model_predictions", "model_report"],
}


def reimport_modules_and_functions():
    import importlib

    for module in imports_1:
        importlib.reload(importlib.import_module(module))
    for module, items in imports_2.items():
        reloaded_module = importlib.import_module(module)
        globals().update({item: getattr(reloaded_module, item) for item in items})


reimport_modules_and_functions()

# %%

# id = ("mp-390", "000_Ti")
# vasp_raw_data = RAWDataVASP(compound=compound)
# processed_vasp_spectra = VASPDataModifier(vasp_raw_data.parameters[id])

# %%

compound = "Ti"
raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)


# %%

# seed = 42
# random.seed(seed)
sample_length = 10
ids = set(raw_data.parameters.keys())
vasp_ids = set(vasp_raw_data.parameters.keys())
common_ids = ids.intersection(vasp_ids)

plt.style.use(["default", "science"])
ids = random.choices(list(common_ids), k=sample_length)
fig, axs = plt.subplots(len(ids), 1, figsize=(6, 3 * len(ids)))
time_corr = []
time_dtw = []
for simulation_type in ["VASP", "FEFF"]:
    raw_data = vasp_raw_data if simulation_type == "VASP" else raw_data
    data_class = VASPData if simulation_type == "VASP" else FEFFData
    for ax, id in zip(axs, ids):
        data = data_class(
            compound=compound,
            params=raw_data.parameters[id],
        )
        # data.transform()

        if data.simulation_type == "FEFF":
            t1 = time.time()
            data.align(VASPData(compound, vasp_raw_data.parameters[id]))
            del_t = time.time() - t1
            time_corr.append(del_t)
            ic(del_t)
        ax.plot(
            data.energy,
            data.spectra,
            label=f"{data.simulation_type}_{id}",
            linestyle="-",
        )

        # doing again for dtw based
        if data.simulation_type == "FEFF":
            data = data_class(
                compound=compound,
                params=raw_data.parameters[id],
            )
            t1 = time.time()
            shift = data_class.dtw_shift(
                data,
                VASPData(compound, vasp_raw_data.parameters[id]),
            )
            del_t = time.time() - t1
            time_dtw.append(del_t)
            ic(del_t)
            data.align_energy(-shift)
            # data.truncate_emperically()
            ax.plot(
                data.energy,
                data.spectra,
                label=f"{data.simulation_type}_{id}_dtw",
                linestyle="--",
            )

        ax.legend()
        ax.sharex(axs[0])

axs[-1].set_xlabel("Energy (eV)", fontsize=18)

plt.suptitle(f"Per-spectra alignment samples: {compound}", fontsize=18)
plt.tight_layout()
# plt.savefig(f"vasp_truncation_examples_{compound}.pdf", bbox_inches="tight", dpi=300)
plt.show()

# ==============================================================================

# %%

# compound = "Ti"
# simulation_type = "FEFF"
# raw_data_class = RAWDataFEFF if simulation_type == "FEFF" else RAWDataVASP
# raw_data = raw_data_class(compound=compound)
# ids = raw_data.parameters.keys()
# ids = tqdm(ids)  # progress bar
# for id in ids:
#     data_class = FEFFData if simulation_type == "FEFF" else VASPData
#     data = data_class(
#         compound=compound,
#         params=raw_data.parameters[id],
#         id=id,
#     )
#     data.save()
#     # plt.plot(data.energy, data.spectra, label=id)
#     # plt.legend()
# # plt.show()

# %%

# %%

compound = "Ti"
id = ("mp-390", "000_Ti")
data_vasp = VASPData(compound).load(id)
data_feff = FEFFData(compound).load(id)

plt.plot(data_vasp.energy, data_vasp.spectra, label="VASP")
plt.plot(data_feff.energy, data_feff.spectra, label="FEFF")
plt.legend()
plt.show()

alignment_shift = data_feff.align(data_vasp)
# data_feff.truncate_emperically()
# data_vasp.truncate_emperically()
plt.plot(data_vasp.energy, data_vasp.spectra, label="VASP")
plt.plot(data_feff.energy, data_feff.spectra, label=f"FEFF_shifted_{alignment_shift}")
plt.show()
# %%

alignment_shift


# %%
def compute_shift(compound, id):
    data_vasp = VASPData(compound).load(id)
    data_feff = FEFFData(compound).load(id)
    alignment_shift = data_feff.align(data_vasp)
    return alignment_shift


compound = "Ti"
data_raw_feff = RAWDataFEFF(compound=compound)
data_raw_vasp = RAWDataVASP(compound=compound)

shift_dict = {}
feff_ids = set(data_raw_feff.parameters.keys())
vasp_ids = set(data_raw_vasp.parameters.keys())
common_ids = feff_ids.intersection(vasp_ids)
common_ids = tqdm(common_ids)  # progress bar

import multiprocessing

with multiprocessing.Pool(processes=8) as pool:
    shift_dict = pool.map(compute_shift, common_ids)

keys = np.array(list(map(lambda x: "_site_".join(x), list(shift_dict.keys()))))
shifts_val = np.array(list(shift_dict.values()))
np.savetxt(
    "shifts.txt",
    np.vstack((keys, shifts_val)).T,
    fmt="%s",
    delimiter="\t",
    header="id_site\talignment_shift",
)

# %%
l = {"a": 1, "b": 2}
list(l.keys())[:1]
# %%


# %%

from scripts.compute_all_shifts import (
    compute_all_shifts,
    save_shifts,
    common_ids,
    compute_shift,
)


def load_shifts(compound):
    file_path = os.path.join(
        "dataset",
        "transfer_learning",
        f"shifts_{compound}.txt",
    )
    shifts_all = np.loadtxt(file_path, delimiter="\t", skiprows=1, dtype=str)
    id_site = shifts_all[:, 0]
    ids_shift = list(map(lambda x: tuple(x.split("_site_")), id_site))
    shifts_val = shifts_all[:, 1].astype(float)
    shifts = dict(zip(ids_shift, shifts_val))
    return shifts


def load_processed_data(compound, id):
    vasp_data = VASPData(compound).load(id)
    feff_data = FEFFData(compound).load(id)
    return vasp_data, feff_data


def interpolate_spectra(data, new_length):
    energy = data.energy
    spectra = data.spectra
    new_energy = np.linspace(energy.min(), energy.max(), new_length)
    new_spectra = np.interp(new_energy, energy, spectra)
    data._energy = new_energy
    data._spectra = new_spectra
    return data


def load_tl_data(compound, id, shifts, FIXED_LENGTH=100):
    vasp_data, feff_data = load_processed_data(compound, id)
    feff_data.align_energy(-shifts[id])
    vasp_data = interpolate_spectra(vasp_data, FIXED_LENGTH)
    feff_data = interpolate_spectra(feff_data, FIXED_LENGTH)
    return vasp_data, feff_data


# %%

compound = "Ti"
ids = common_ids(compound)
id = random.choice(ids)
vasp_data, feff_data = load_processed_data(compound, id)
plt.plot(vasp_data.energy, vasp_data.spectra, label="VASP")
plt.plot(feff_data.energy, feff_data.spectra, label="FEFF")
plt.legend()


# %%

compound = "Cu"
ids = common_ids(compound)

feff_lenghts = [len(load_processed_data(compound, id)[1]) for id in ids]

vasp_lenghts = [len(load_processed_data(compound, id)[0]) for id in ids]


# %%

compound = "Ti"
fig = plt.figure(figsize=(8, 6))
feff_raw = RAWDataFEFF(compound=compound)
vasp_raw = RAWDataVASP(compound=compound)

# %%

seed = 1
common_ids = common_ids(compound)
random.seed(seed)
id = random.choice(list(common_ids))
feff_data = FEFFData(compound, feff_raw.parameters[id])
vasp_data = VASPData(compound, vasp_raw.parameters[id])

# def align(self, emperical_offset=5114.08973):
#     theoretical_offset = (self.e_core - self.e_cbm) + (self.E_ch - self.E_GS)
#     super().align_energy(theoretical_offset)
#     super().align_energy(emperical_offset)
#     return self

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(feff_data.energy, feff_data.spectra, label=f"FEFF_{id}")
axs[0].set_title("FEFF")
axs[0].set_ylabel("Spectra", fontsize=20)
axs[0].set_xlabel("Energy (eV)", fontsize=20)
axs[0].sharey(axs[1])

axs[1].plot(vasp_data.energy, vasp_data.spectra, label=f"VASP_{id}")
axs[1].set_title("VASP")
axs[1].set_xlabel("Energy (eV)", fontsize=20)

plt.xlabel("Energy (eV)", fontsize=20)
plt.suptitle(f"FEFF spectra: {id}", fontsize=20)
plt.savefig(f"feff_vasp_spectra_{id}.pdf", bbox_inches="tight", dpi=300)


# %%

plt.style.use(["default", "science"])
fig = plt.figure(figsize=(8, 6))
plt.hist(feff_lenghts, bins=20, label="FEFF", color="green", edgecolor="tab:green")
plt.title(f"FEFF spectra lengths: {compound}", fontsize=20)
plt.xlabel("Length", fontsize=20)
plt.ylabel("Count", fontsize=20)
plt.legend()
plt.savefig(f"feff_spectra_lengths_{compound}.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%

compound = "Cu"
ids = common_ids(compound)
shifts = load_shifts(compound)

# %%
plt.style.use(["default", "science"])
fig = plt.figure(figsize=(8, 6))
id = random.choice(ids)
feff_data, vasp_data = load_tl_data(id, shifts)
plt.plot(vasp_data.energy, vasp_data.spectra, label="VASP")
plt.plot(feff_data.energy, feff_data.spectra, label="FEFF")
plt.legend()
plt.xlabel("Energy (eV)", fontsize=20)
plt.ylabel("Spectra", fontsize=20)
plt.title(f"FEFF-VASP TL data: {id}", fontsize=20)
plt.savefig(f"tl_data_{id}.pdf", bbox_inches="tight", dpi=300)
plt.show()


# %%

diffs = []
for id in ids:
    vasp_data, feff_data = load_tl_data(id, shifts)
    diffs.append(vasp_data.spectra - feff_data.spectra)

# %%
plt.style.use(["default", "science"])
fig = plt.figure(figsize=(8, 6))
plt.plot(np.mean(diffs, axis=0), label=f"mean_tl_diff_{compound}")
plt.fill_between(
    np.arange(len(np.mean(diffs, axis=0))),
    np.mean(diffs, axis=0) + np.std(diffs, axis=0),
    np.mean(diffs, axis=0) - np.std(diffs, axis=0),
    alpha=0.2,
)
plt.xlabel("Energy (eV)", fontsize=20)
plt.ylabel("Spectra difference", fontsize=20)
plt.title(f"FEFF-VASP TL data difference: {compound}", fontsize=20)
plt.savefig(f"tl_data_diff_{compound}.pdf", bbox_inches="tight", dpi=300)

# %%

# load csv data
cu2o_data = np.loadtxt(
    os.path.join("dataset", "cu2o_experiment", "cu2o.csv"), delimiter=","
)
plt.plot(cu2o_data[:, 0], cu2o_data[:, 1])

# %%


def get_common_ids(vasp_raw_data, feff_raw_data):
    feff_ids = set(feff_raw_data.parameters.keys())
    vasp_ids = set(vasp_raw_data.parameters.keys())
    common_ids = feff_ids.intersection(vasp_ids)
    return common_ids


def random_sample(vasp_raw_data, feff_raw_data):
    common_ids = get_common_ids(vasp_raw_data, feff_raw_data)
    id = random.choice(list(common_ids))
    vasp_data = VASPData(compound, vasp_raw_data.parameters[id])
    feff_data = FEFFData(compound, feff_raw_data.parameters[id])
    return vasp_data, feff_data


def resample_spectra(vasp_data, feff_data, sample_length):
    new_energy = np.linspace(vasp_data.energy[0], vasp_data.energy[-1], sample_length)

    new_vasp_data = deepcopy(vasp_data)
    new_vasp_data.spectra = np.interp(
        new_energy,
        new_vasp_data.energy,
        new_vasp_data.spectra,
    )
    new_vasp_data.energy = new_energy

    new_feff_data = deepcopy(feff_data)
    new_feff_data.spectra = np.interp(
        new_energy,
        new_feff_data.energy,
        new_feff_data.spectra,
    )
    new_feff_data.energy = new_energy

    return new_vasp_data, new_feff_data


def plot_spectras(vasp_data, feff_data):
    plt.plot(vasp_data.energy, vasp_data.spectra)
    plt.plot(feff_data.energy, feff_data.spectra)


# %%

compound = "Cu"
vasp_raw_data = RAWDataVASP(compound=compound)
feff_raw_data = RAWDataFEFF(compound=compound)

# %%

common_ids = get_common_ids(vasp_raw_data, feff_raw_data)
seed = 1
random.seed(seed)
id = random.choice(list(common_ids))
feff_data = FEFFData(compound, feff_raw_data.parameters[id])
vasp_data = VASPData(compound, vasp_raw_data.parameters[id])
plt.plot(feff_data.energy, feff_data.spectra, label=f"FEFF_{id}")
plt.plot(vasp_data.energy, vasp_data.spectra, label=f"VASP_{id}")
plt.legend()

# %%
seed = 42
random.seed(seed)
vasp_data, feff_data = random_sample(vasp_raw_data, feff_raw_data)
v0, f0 = vasp_data, feff_data
v1, f1 = resample_spectra(vasp_data, feff_data, sample_length)

plt.plot(v0.energy, v0.spectra, label="VASP", c="red")
plt.scatter(v1.energy, v1.spectra, label="VASP_resampled")


# %%
ic(v1.energy)


def spectral_length_histogram(compound, vasp_raw_data, feff_raw_data):
    feff_ids = set(feff_raw_data.parameters.keys())
    vasp_ids = set(vasp_raw_data.parameters.keys())
    common_ids = feff_ids.intersection(vasp_ids)

    feff_data_length = []
    vasp_data_length = []
    for id in common_ids:
        # id = random.choice(list(common_ids))
        feff_data = FEFFData(compound, feff_raw_data.parameters[id])
        # feff_data = feff_data.truncate_emperically()
        vasp_data = VASPData(compound, vasp_raw_data.parameters[id])
        # vasp_data = vasp_data.truncate_emperically()

        feff_data_length.append(len(feff_data))
        vasp_data_length.append(len(vasp_data))
    feff_data_length = np.array(feff_data_length)
    vasp_data_length = np.array(vasp_data_length)

    # HISTOGRAM OF SPECTRA LENGTHS
    plt.style.use(["default", "science"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].hist(
        feff_data_length, bins=20, label="FEFF", color="green", edgecolor="green"
    )
    axs[0].set_title("FEFF")
    axs[0].set_xlabel("Length", fontsize=20)
    axs[0].set_ylabel("Count", fontsize=20)
    axs[0].sharey(axs[1])
    axs[1].hist(vasp_data_length, bins=20, label="VASP", color="blue", edgecolor="blue")
    axs[1].set_title("VASP")
    axs[1].set_xlabel("Length", fontsize=20)
    axs[1].set_ylabel("Count", fontsize=20)
    highlight_feff = np.median(feff_data_length)
    highlight_vasp = np.median(vasp_data_length)
    highlight_tick(axs[0], highlight_feff)  # horizontal axis
    highlight_tick(axs[1], highlight_vasp)  # horizontal axis
    # vertical line
    axs[0].axvline(
        highlight_feff, color="red", linestyle="--", linewidth=2, label="median"
    )
    axs[1].axvline(
        highlight_vasp, color="red", linestyle="--", linewidth=2, label="median"
    )
    axs[0].legend()
    axs[1].legend()
    plt.suptitle(f"FEFF-VASP spectra lengths: {compound}", fontsize=20)
    plt.savefig(
        f"feff_vasp_spectra_lengths_{compound}.pdf", bbox_inches="tight", dpi=300
    )


spectral_length_histogram(compound, vasp_raw_data, feff_raw_data)


# %%
