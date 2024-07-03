# %%

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import circmean

from config.defaults import cfg
from EXAFS import EXAFS_compound


def calculate_average_exafs(exafs_list):
    R = exafs_list[0][0]  # Assuming R is the same for all spectra
    chi_R_abs_list = [np.abs(chi_R) for _, chi_R in exafs_list]
    avg_chi_R = np.mean(chi_R_abs_list, axis=0)
    std_chi_R = np.std(chi_R_abs_list, axis=0)
    return R, avg_chi_R, std_chi_R


def plot_exafs(exafs_data, compounds, simulation_types, model_names):
    COLS = len(compounds)
    ROWS = len(model_names) + 1  # Add one row for the average plot
    BIN_COUNT = 100
    CMAP = "tab10"
    ALPHA = 1
    LINEWIDTH = 0.5
    FONTSIZE = 22

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
                    r"median({r'$R_{peak}$'}) = {median_peak:.3f} $\AA$",
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


# %%


# DEFINE VARIABLES
# simulation_types = ["FEFF", "VASP"]
models = ["simulation", "universal", "expert", "tuned_universal"]
vasp_compounds = ["Cu", "Ti"]
vasp_models = ["simulation", "expert", "tuned_universal"]

# # VASP PARAMETERS
# simulation_type = "VASP"
# compounds = vasp_compounds
# models = vasp_models

# FEFF PARAMETERS
simulation_type = "FEFF"
compounds = cfg.compounds
models = ["simulation", "universal", "expert", "tuned_universal"]


exafs_data = {}
for compound in compounds:
    exafs_data[compound] = {}
    exafs_data[compound][simulation_type] = {}
    for model_name in models:
        exafs_data[compound][simulation_type][model_name] = EXAFS_compound(
            compound, simulation_type, model_name
        )


# # Plot EXAFS data
# plot_exafs(exafs_data, compounds, simulation_types, model_names)

# %%

PLOT_RESIDUALS = False

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(3, len(compounds), hspace=0, wspace=0.0)
ax_all = gs.subplots(
    sharex=True,
    sharey=True,
    subplot_kw={"projection": "polar"},
)
plt.style.use(["default", "science"])


for COMPOUND, axs in zip(compounds, ax_all.T):

    if PLOT_RESIDUALS:
        baseline_model = "simulation"
        base_line_exafs = np.array(
            exafs_data[COMPOUND][simulation_type][baseline_model]
        )
        baseline_theta = np.angle(base_line_exafs[:, 1])
        baseline_r = np.abs(base_line_exafs[:, 1])

    for ax, PRED_NAME in zip(axs, models):
        print(f"Plotting {PRED_NAME} for {COMPOUND} - {simulation_type}")

        exafs = np.array(exafs_data[COMPOUND][simulation_type][PRED_NAME])
        exafs_freqs = exafs[:, 0]
        exafs_values = exafs[:, 1]

        print(f"Lenth of exafs_values for {COMPOUND} = {len(exafs_values)}")

        r = np.abs(exafs_values)
        theta = np.angle(exafs_values)

        if PLOT_RESIDUALS:
            r -= baseline_r
            theta -= baseline_theta

        scatter = ax.scatter(
            theta,
            r,
            s=1,
            c=np.real(exafs_freqs),
            cmap="jet",
            norm=LogNorm(),
            # zorder=2,
        )

        ax.set_yticklabels([])
        ax.set_yscale("log")
        ax.set_title(f"{PRED_NAME} - {COMPOUND} - {simulation_type}")
        plt.tight_layout()
        print("\n")
        break
    print("\n")
    break

filename = "exafs_polar_residuals.pdf" if PLOT_RESIDUALS else "exafs_polar.pdf"
plt.savefig(filename, bbox_inches="tight", dpi=300)

# %%


# %%
