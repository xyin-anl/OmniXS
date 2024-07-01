# %%
import scienceplots
import os
import numpy as np
import torch
from EXAFS import EXAFS
from src.models.trained_models import Trained_FCModel
from src.data.ml_data import load_xas_ml_data, DataQuery
import pickle
from config.defaults import cfg
from matplotlib import pyplot as plt
import warnings


# %%


def generate_and_save_exafs(
    compounds, simulation_types, model_names, force_recalculate=False
):
    """
    Generate EXAFS data for all compounds, simulation types, and model types.
    Save the data to a file and return it.

    :param compounds: List of compound names
    :param simulation_types: List of simulation types
    :param model_names: List of model names
    :param force_recalculate: If True, recalculate data even if file exists
    :return: Dictionary containing EXAFS data
    """

    # filename = "exafs_data.pkl"
    filename = cfg.paths.cache.exafs

    if not force_recalculate and os.path.exists(filename):
        warnings.warn(f"Loading existing EXAFS data from {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)

    exafs_data = {}

    for compound in compounds:
        exafs_data[compound] = {}
        for simulation_type in simulation_types:
            exafs_data[compound][simulation_type] = {}

            data = load_xas_ml_data(DataQuery(compound, simulation_type)).test
            spectras = data.y
            features = data.X

            for model_name in model_names:
                if model_name == "simulation":
                    preds = spectras
                elif model_name == "universal":
                    preds = (
                        Trained_FCModel(
                            DataQuery("ALL", simulation_type), name="universal_tl"
                        )
                        .model(torch.tensor(features))
                        .detach()
                        .numpy()
                    )
                elif model_name == "expert":
                    preds = Trained_FCModel(
                        DataQuery(compound, simulation_type), name="per_compound_tl"
                    ).predictions
                elif model_name == "tuned_universal":
                    preds = Trained_FCModel(
                        DataQuery(compound, simulation_type), name="ft_tl"
                    ).predictions

                exafs_list = []
                for spectra in preds:
                    exafs = EXAFS(
                        spectra, compound=compound, simulation_type=simulation_type
                    )
                    R, chi_R = exafs.chi_k_fft
                    exafs_list.append((R, chi_R))  # Store full complex data

                exafs_data[compound][simulation_type][model_name] = exafs_list

    with open(filename, "wb") as f:
        pickle.dump(exafs_data, f)

    return exafs_data


def calculate_average_exafs(exafs_list):
    R = exafs_list[0][0]  # Assuming R is the same for all spectra
    chi_R_abs_list = [np.abs(chi_R) for _, chi_R in exafs_list]
    avg_chi_R = np.mean(chi_R_abs_list, axis=0)
    std_chi_R = np.std(chi_R_abs_list, axis=0)
    return R, avg_chi_R, std_chi_R


def plot_exafs(exafs_data, compounds, simulation_types, model_names):
    COLS = len(compounds)
    ROWS = len(model_names) + 1  # Add one row for the average plot

    fig = plt.figure(figsize=(6 * COLS, 8 * ROWS))
    gs = fig.add_gridspec(ROWS, COLS, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    colors = plt.get_cmap(CMAP)(np.arange(len(model_names)))
    colors_dict = dict(zip(model_names, colors))

    for col, compound in enumerate(compounds):
        for simulation_type in simulation_types:
            # Plot individual model data
            for row, model in enumerate(model_names):
                exafs_list = exafs_data[compound][simulation_type][model]

                for R, chi_R in exafs_list:
                    chi_R_abs = np.abs(chi_R)
                    axs[row, col].plot(
                        R,
                        chi_R_abs,
                        color=colors_dict[model],
                        alpha=ALPHA,
                        linewidth=LINEWIDTH,
                    )
                    axs[row, col].set_xlabel(r"$R$ ($\AA$)", fontsize=FONTSIZE)
                    axs[row, col].set_xlim(0, 4)
                    axs[row, col].set_yscale("symlog", linthresh=1e-3)
                    axs[row, col].set_yticks([0, 1e-3, 1e-2, 1e-1, 1])
                    axs[row, col].set_yticklabels([0, 1e-3, 1e-2, 1e-1, 1])

                ax_inset = axs[row, col].inset_axes([0.7, 0.7, 0.3, 0.3])
                peaks = [
                    np.abs(chi_R).argmax() * (R[1] - R[0]) for _, chi_R in exafs_list
                ]
                ax_inset.hist(
                    peaks, bins=BIN_COUNT, color=colors_dict[model], density=True
                )
                ax_inset.set_xlabel(r"$R_{peak}$ ($\AA$)", fontsize=FONTSIZE * 0.5)
                ax_inset.set_xlim(0, 1)

                median_peak = np.median(peaks)
                axs[row, col].text(
                    0.05,
                    0.001,
                    f"median({r'$R_{peak}$'}) = {median_peak:.3f} $\AA$",
                    fontsize=FONTSIZE * 0.5,
                )
                axs[row, col].vlines(
                    median_peak,
                    0,
                    np.max(chi_R_abs),
                    color=colors_dict[model],
                    linestyle="--",
                    alpha=1,
                )
                axs[row, col].set_ylim(0, None)

            # Plot average EXAFS with std
            row = len(model_names)  # Last row
            for model in model_names:
                exafs_list = exafs_data[compound][simulation_type][model]
                R, avg_chi_R, std_chi_R = calculate_average_exafs(exafs_list)
                axs[row, col].plot(
                    R,
                    avg_chi_R,
                    color=colors_dict[model],
                    label=model,
                    linewidth=LINEWIDTH * 2,
                )

                # axs[row, col].fill_between(
                #     R,
                #     avg_chi_R - std_chi_R,
                #     avg_chi_R + std_chi_R,
                #     color=colors_dict[model],
                #     alpha=0.3,
                # )

            axs[row, col].set_xlabel(r"$R$ ($\AA$)", fontsize=FONTSIZE)
            axs[row, col].set_xlim(0, 4)
            axs[row, col].set_yscale("symlog", linthresh=1e-3)
            axs[row, col].set_yticks([0, 1e-3, 1e-2, 1e-1, 1])
            axs[row, col].set_yticklabels([0, 1e-3, 1e-2, 1e-1, 1])
            axs[row, col].legend(fontsize=FONTSIZE * 0.6)

            # Set y-axis label only for the leftmost column
            if col == 0:
                for row in range(ROWS):
                    axs[row, col].set_ylabel(r"$|\chi(R)|$", fontsize=FONTSIZE * 0.8)
            else:
                for row in range(ROWS):
                    axs[row, col].set_yticklabels([])

            # Set title only for first row
            axs[0, col].set_title(
                f"{compound} - {simulation_type}", fontsize=FONTSIZE * 0.9
            )

    # Set overall title for rows (model types)
    for row, model in enumerate(model_names + ["Average"]):
        fig.text(
            0.01,
            (1 - (row + 0.5) / ROWS),
            model,
            ha="left",
            va="center",
            rotation="vertical",
            fontsize=FONTSIZE * 1.2,
        )

    plt.suptitle(f"EXAFS - {simulation_types[0]}", fontsize=FONTSIZE * 1.5, y=1.02)
    fig.tight_layout()
    plt.savefig(f"exafs_{simulation_types[0]}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


plt.style.use(["default", "science"])
BIN_COUNT = 100
CMAP = "tab10"
ALPHA = 1
LINEWIDTH = 0.5
FONTSIZE = 22
compounds = cfg.compounds
simulation_types = ["FEFF"]
model_names = ["simulation", "universal", "expert", "tuned_universal"]

# Generate and save EXAFS data (or load if already exists)
exafs_data = generate_and_save_exafs(
    compounds,
    simulation_types,
    model_names,
    force_recalculate=False,
)

# Plot EXAFS data
plot_exafs(exafs_data, compounds, simulation_types, model_names)

# If you want to load existing data later
# plot_exafs(exafs_data, compounds, simulation_types, model_names)

# %%
