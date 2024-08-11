# %%

# kde
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import Trained_FCModel, MeanModel
from config.defaults import cfg

# %%

mse_diffs = {}
for compound in cfg.compounds:
    expert_model = Trained_FCModel(DataQuery(compound, "FEFF"), name="per_compound_tl")
    tuned_univ_model = Trained_FCModel(DataQuery(compound, "FEFF"), name="ft_tl")
    mse_diffs[compound] = (
        expert_model.mse_per_spectra - tuned_univ_model.mse_per_spectra
    )

for compound in ["Ti", "Cu"]:
    expert_model = Trained_FCModel(DataQuery(compound, "VASP"), name="per_compound_tl")
    tuned_univ_model = Trained_FCModel(DataQuery(compound, "VASP"), name="ft_tl")
    mse_diffs[compound + "_VASP"] = (
        expert_model.mse_per_spectra - tuned_univ_model.mse_per_spectra
    )

# %%


fig = plt.figure(figsize=(8, 10))
plt.style.use(["default", "science"])
gs = fig.add_gridspec(4, 2, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=True)
i = 0
FONTSIZE = 18
for ax, (compound, mse_diff) in zip(axs.ravel(), mse_diffs.items()):

    i += 1

    ax.set_xlim(-0.002, 0.004) if compound != "Ti_VASP" else ax.set_xlim(-0.015, 0.04)

    BIN_WIDTH = 0.0002

    if "VASP" in compound:
        compound = compound.replace("_VASP", "")
        baseline_median = np.median(
            MeanModel(DataQuery(compound, "VASP")).mse_per_spectra
        )
    else:
        baseline_median = np.median(
            MeanModel(DataQuery(compound, "FEFF")).mse_per_spectra
        )

    data = mse_diff

    ax.hist(
        data,
        density=True,
        bins=np.arange(min(data), max(data) + BIN_WIDTH, BIN_WIDTH),
        # bins=100,
        # black lines
        # histtype="step",
        color=plt.cm.tab10(i),
        alpha=0.5,
        # line color and style
        edgecolor=plt.cm.tab10(i),
        # cumulative=True,
    )

    ax.vlines(
        0,
        0,
        1e6,
        color="black",
        linestyle="--",
        label=r"$\Delta$MSE = 0",
        alpha=0.5,
        linewidth=1,
    )
    
    ax.set_yscale("log")
    ax.set_ylim(1e1, 5e3)

for i in range(4):
    axs[i, 0].set_ylabel("Density", fontsize=FONTSIZE)
    axs[i, 0].set_yticks([1e0, 1e1, 1e2, 1e3])

for i in range(2):
    axs[3, i].set_xlabel(r"$\Delta$MSE", fontsize=FONTSIZE)
    axs[3, i].set_xlabel(r"$\Delta$MSE", fontsize=FONTSIZE)
    axs[3, i].legend(fontsize=FONTSIZE * 0.8)


plt.tight_layout()

# %%
