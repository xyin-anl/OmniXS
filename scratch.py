# %%
# %load_ext autoreload
# %autoreload 2

# from scripts.plots_model_report import plot_residue_histogram
import warnings
from scipy.stats import cauchy
import numpy as np
from scipy.signal import convolve

import numpy as np
from utils.src.plots.highlight_tick import highlight_tick
import yaml
import scienceplots
from src.model_report import linear_model_predictions
from src.model_report import model_report
import lightning as pl
import ast
from scripts.pca_plots import plot_pcas, linear_fit_of_pcas
import pandas as pd
import seaborn as sns
from src.xas_data import XASData
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scienceplots
from scripts.plots_model_report import (
    plot_residue_quartiles,
    # plot_residue_histogram,
    plot_residue_cv,
    plot_residue_heatmap,
    plot_predictions,
    heatmap_of_lines,
)
from utils.src.lightning.pl_module import PLModule
import torch
from pprint import pprint
from utils.src.optuna.dynamic_fc import PlDynamicFC
from src.ckpt_predictions import get_optimal_fc_predictions
from typing import TypedDict, Union, Tuple
from src.xas_data_raw import RAWData


# %%

## MISSING DATA
# for compound in ["Ti", "Cu"]:
#     data = RAWData(compound, "VASP")
#     print(
#         f"RAW VASP data for {compound}: \
#             {len(data)}, {len(data.missing_data)} (missing)"
#     )
#     with open(f"missing_VASP_data_{compound}.txt", "w") as f:
#         for d in data.missing_data:
#             f.write(f"{d}\n")

# %%

# compound = "Ti"
compound = "Cu"
simulation_type = "VASP"
data = RAWData(compound, simulation_type)


id = next(iter(data.parameters))
single_spectra = data.parameters[id]["mu"]
energy_full = single_spectra[:, 0]
spectra_full = single_spectra[:, 1]
e_core = data.parameters[id]["e_core"]
e_cbm = data.parameters[id]["e_cbm"]
E_ch = data.parameters[id]["E_ch"]
E_GS = data.parameters[id]["E_GS"]
volume = data.parameters[id]["volume"]


# E_min is based on theory with arbitary min_offset
min_offset = 5
theory_min = (e_cbm - e_core) - min_offset
energy_min_idx = np.where(energy_full >= theory_min)[0][0]
energy_min = energy_full[energy_min_idx]
# E_max is last non-zero value
energy_max_idx = np.where(spectra_full != 0)[0][-1]
energy_max = energy_full[energy_max_idx]
# spectra and energy are now trimmed
spectra = spectra_full[energy_min_idx : energy_max_idx + 1]
energy = energy_full[energy_min_idx : energy_max_idx + 1]

plot_legend_non_zero = r"Non-zero E range\\ "
plot_legend_non_zero += (
    r"$E_{\text{min}} = e_{\text{cbm}} - e_{\text{core}} - \Delta E_0$ \\ "
)
plot_legend_non_zero += (
    r"$E_{\text{min}} ="
    + f"{e_cbm:.2f} - {e_core:.2f} - {min_offset} "
    + r"\ \text{eV}$\\ "
)
plot_legend_non_zero += (
    r"$E_{\text{min}} = " + f"{energy_min:.2f}" + r"\ \text{eV}$" + " (available)\n"
)
plot_legend_non_zero += r"$E_{\text{max}} = " + f"{energy_max:.2f}" + r"\ \text{eV}$"

## SCALING
omega = spectra * energy
big_omega = volume
alpha = 1 / 137  # TODO: ask if it is fine-structure constant or something else
spectra_scaled = (omega * big_omega) / alpha  # ASK if alpha is mul or div


plot_legend_scaling = r"Scaling: \\ "
plot_legend_scaling += r"$\omega' = \omega \cdot \Omega / \alpha$ \\ "
plot_legend_scaling += r"$\omega = y \cdot E$ \\ "
plot_legend_scaling += f"${{\\Omega}}= {big_omega:.2f}$ \n "
plot_legend_scaling += f"${{\\alpha}}= {alpha:.3f}$"


## BRODENING
# Source: https://github.com/AI-multimodal/Lightshow/blob/mc-broadening-revisions/lightshow/postprocess/broaden.py
# fixed gamma by half
def lorentz_broaden(x, xin, yin, gamma):
    x1, x2 = np.meshgrid(x, xin)
    dx = xin[-1] - xin[0]
    return np.dot(cauchy.pdf(x1, x2, gamma).T, yin) / len(xin) * dx


Gamma = 0.89
gamma = Gamma/2
broadened_amplitude = lorentz_broaden(
    energy,
    energy,
    spectra_scaled,
    gamma=gamma,  # TODO: mention this
)
plot_legend_broadening = r"Broadening: \\ "
plot_legend_broadening += r"Lorentzian broadening: PDF of Cauchy distribution \\ "
plot_legend_broadening += r"$f(x, k) = \frac{1}{2^{k/2-1} \gamma \left( k/2 \right)}$"
plot_legend_broadening += r"$x^{k-1} \exp \left( -x^2/2 \right)$ \\ "
plot_legend_broadening += r"$\gamma = \Gamma / 2$ \\ "
plot_legend_broadening += r"$\Gamma = 0.89$"


## ALIGNMENT
offset = (e_core - e_cbm) + (E_ch - E_GS)
plot_legend_alignment = r"Alignment: \\ "
plot_legend_alignment += (
    r"$\Delta E = (\epsilon_{\text{core}} - \epsilon_{\text{cbm}})$"
)
plot_legend_alignment += r"$+ (E_{\text{ch}} - E_{\text{GS}})$ \\ "
plot_legend_alignment += (
    f"$\\Delta E = ({e_core:.2f} - {e_cbm:.2f}) + ({E_ch:.2f} - {E_GS:.2f})$ \n "
)
plot_legend_alignment += r"$\Delta E = $" f"{offset:.2f} \n"
# plot_legend_alignment += r"$e_{{\text{{core}}}}$" + f"={e_core:.2f}\n"
energy_aligned = energy + offset


plt.style.use(["science", "vibrant"])
fig, ax = plt.subplots(6, 1, figsize=(15, 15))
ax[0].plot(energy_full, spectra_full, label="Raw")
ax[1].plot(energy, spectra, label=plot_legend_non_zero)
ax[2].plot(energy, spectra_scaled, label=plot_legend_scaling)
ax[3].plot(energy, broadened_amplitude, label=plot_legend_broadening)
ax[4].plot(energy_aligned, broadened_amplitude, label=plot_legend_alignment)
ax[5].plot(energy_aligned, broadened_amplitude, label="aligned, scaled, broadened")
ax[5].plot(energy_aligned, spectra_scaled, label="aligned, scaled, not broadened")
# ax[3].sharex(ax[2])
# ax[4].sharex(ax[2])
for axis in ax:
    axis.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=13)
fig.suptitle(
    f"Scaling, Alignment, Broadening: {simulation_type} {compound} {id}",
    fontsize=18,
    color="red",
)
plt.tight_layout()
plt.savefig(
    f"scaling_alignment_broadening_{simulation_type}_{compound}_{id}.pdf",
    dpi=300,
)


