# %%
from scipy.stats import gaussian_kde
import torch
from EXAFS import EXAFSSpectrum
from src.models.trained_models import Trained_FCModel
from scipy.signal import tukey
import numpy as np
import scienceplots
from matplotlib import pyplot as plt

from src.data.ml_data import load_xas_ml_data, DataQuery

plt.style.use(["default", "science"])

# %%

compound = "Cu"
simulation_type = "VASP"
data = load_xas_ml_data(DataQuery(compound, simulation_type)).test
spectras = data.y
features = data.X
pred_expert = Trained_FCModel(
    DataQuery(compound, "FEFF"), name="per_compound_tl"
).predictions

pred_universal = (
    Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
    .model(torch.tensor(features))
    .detach()
    .numpy()
)
pred_tuned_universal = Trained_FCModel(
    DataQuery(compound, simulation_type), name="ft_tl"
).predictions

ALPHA = 1
LINEWIDTH = 0.1
model_preds = {
    "simulation": spectras,
    # "universal": pred_universal,
    "expert": pred_expert,
    "tuned_universal": pred_tuned_universal,
}
CMAP = "tab10"
BIN_COUNT = 50
FONTSIZE = 18
colors = plt.get_cmap(CMAP)(np.arange(len(model_preds)))
colors_dict = dict(zip(model_preds.keys(), colors))


exafsFigure = plt.figure(figsize=(6, 8))
gs = exafsFigure.add_gridspec(len(model_preds), 1, hspace=0, wspace=0)
ax = gs.subplots(sharex=True, sharey=False)
medians = {}
for i, (model, preds) in enumerate(model_preds.items()):
    for spectra in preds:
        exafs = EXAFSSpectrum(spectra, compound=compound, simulation_type="FEFF")
        R, chi_R = exafs.chi_k2_fft
        chi_R = np.abs(chi_R)
        ax[i].plot(R, chi_R, color=colors_dict[model], alpha=ALPHA, linewidth=LINEWIDTH)
        ax[i].set_xlabel(r"$R$ ($\AA$)", fontsize=FONTSIZE)
        ax[i].set_ylabel(f"{model} \n" + r"$\chi(R)$", fontsize=FONTSIZE * 0.8)
        # ax[i].set_title(model, fontsize=FONTSIZE)
        ax[i].set_xlim(0, 4)
        ax[i].set_yscale("symlog", linthresh=1e-3)
        # ticks at 0, 3,2,1 only
        ax[i].set_yticks([0, 1e-3, 1e-2, 1e-1, 1])
        ax[i].set_yticklabels([0, 1e-3, 1e-2, 1e-1, 1])

    # inset with peak location histogram
    ax_inset = ax[i].inset_axes([0.7, 0.7, 0.3, 0.3])
    peaks = []
    for spectra in preds:
        exafs = EXAFSSpectrum(spectra, compound=compound, simulation_type="FEFF")
        peaks.append(exafs.chi_r_peak()[0])
    ax_inset.hist(peaks, bins=BIN_COUNT, color=colors_dict[model], density=True)
    ax_inset.set_xlabel(r"$R_{peak}$ ($\AA$)", fontsize=FONTSIZE * 0.5)
    ax_inset.set_xlim(0, 1)

    # add most frequent of peak to main plot as vline with label and legend at bottom left
    median_peak = np.median(peaks)
    ax[i].text(
        0.05,
        0.001,
        f"median({r'$R_{peak}$'}) = {median_peak:.3f} $\AA$",
    )
    ax[i].vlines(
        median_peak,
        0,
        np.max(chi_R),
        color=colors_dict[model],
        linestyle="--",
        alpha=0.5,
    )
    ax[i].set_ylim(0, None)

    medians[model] = median_peak
exafsFigure.suptitle(
    f"EXAFS of {compound}_{simulation_type}", fontsize=FONTSIZE, x=0.5, y=0.925
)
exafsFigure.tight_layout()
# exafsFigure.savefig(f"exafs_{compound}.pdf", bbox_inches="tight", dpi=300)
exafsFigure.show()

# %%


# make sepeate plot for histogram
exafsPeakFig = plt.figure(figsize=(8, 8))
gs = exafsPeakFig.add_gridspec(2, 1, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)

base_model = "simulation"
kdes = {}
for model, preds in model_preds.items():
    peaks = []
    for spectra in preds:
        exafs = EXAFSSpectrum(spectra, compound=compound, simulation_type="FEFF")
        peaks.append(exafs.chi_r_peak()[0])

    kde = gaussian_kde(peaks)
    kdes[model] = kde

    x = np.linspace(min(peaks), max(peaks), 100)
    axs[0].plot(
        x,
        kde(x),
        color=colors_dict[model],
        linestyle="-",
        label=model,
    )
    axs[0].set_xlabel(r"$R_{peak}$ ($\AA$)", fontsize=FONTSIZE)
    axs[0].set_ylabel("Density", fontsize=FONTSIZE)
    axs[0].legend(fontsize=0.8 * FONTSIZE)

    # diff
    if model == base_model:
        continue
    label = f"{model} - {base_model}"
    # add mse as well
    label += f" (MSE: {np.mean((kde(x) - kdes[base_model](x))**2):.2f})"
    axs[1].plot(
        x,
        kde(x) - kdes[base_model](x),
        color=colors_dict[model],
        label=label,
    )
    axs[1].set_xlabel(r"$R_{peak}$ ($\AA$)", fontsize=FONTSIZE)
    axs[1].set_ylabel("Density", fontsize=FONTSIZE)
    axs[1].legend(fontsize=0.8 * FONTSIZE)
    axs[1].hlines(0, min(peaks), max(peaks), color="gray", linestyle="--", alpha=0.5)

plt.suptitle(
    f"Peak location distribution of {compound}_{simulation_type}", fontsize=FONTSIZE
)
exafsPeakFig.tight_layout()
# exafsPeakFig.savefig(f"exafs_peak_dist_{compound}.pdf", bbox_inches="tight", dpi=300)
exafsPeakFig.show()

# %%

# combine two figures and put side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].imshow(exafsFigure.get_figure().canvas.buffer_rgba(), aspect="auto")
axs[0].axis("off")
axs[1].imshow(exafsPeakFig.get_figure().canvas.buffer_rgba(), aspect="auto")
axs[1].axis("off")
fig.tight_layout()
fig.savefig(
    f"exafs_combined_{compound}_{simulation_type}.pdf", bbox_inches="tight", dpi=300
)


# # %%


# %%
