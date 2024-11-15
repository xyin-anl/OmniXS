# %%
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from omnixas.data import DataTag
from omnixas.core.constants import ElementsFEFF, ElementsVASP, SpectrumType
from omnixas.model.trained_model import ComparisonMetrics, MeanModel, ModelTag
from omnixas.scripts.plots.scripts import CompareAllExpertAndTuned


def format_twin_ticks(twin_ax: plt.Axes, train_avg: np.ndarray) -> None:
    """Format twin axis ticks using nice intervals."""
    y_max = train_avg.max()
    # Scale to get numbers around 1
    scale = 10**3  # Convert to milli-units

    # Set two tick marks at 0.5 and 1.0 of scaled max
    max_scaled = np.ceil(y_max * scale)
    if max_scaled < 0.75:
        ticks = np.array([0.25, 0.5]) * max_scaled
    else:
        ticks = np.array([0.5, 1.0]) * max_scaled

    # Convert back to original scale for plotting
    plot_ticks = ticks / scale
    twin_ax.set_yticks(plot_ticks)
    # Format as clean integers since we're using a multiplier in the label
    twin_ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
    # Make sure the axis extends slightly above the highest tick
    twin_ax.set_ylim(0, plot_ticks[-1] * 1.2)


def plot_element_panel(
    ax: plt.Axes,
    element: str,
    spectrum_type: SpectrumType,
    idx: int,
    compound_colors: dict,
    energy_points: np.ndarray,
    comparison_metrics: dict,
    fontsize: int,
) -> None:
    """Plot a single panel for an element."""
    tag = ModelTag(name="expertXAS", element=element, type=spectrum_type)

    # Get residual differences and baseline
    differences = comparison_metrics[tag].residual_diff
    avg_residues = np.mean(differences, axis=0)
    train_avg = MeanModel.from_data_tag(
        DataTag(element=element, type=spectrum_type)
    ).metrics.predictions[0]
    baseline = train_avg.mean()
    avg_residues = avg_residues / baseline * 100

    # Calculate win rate
    win_rate = np.mean(np.mean(differences, axis=0) > 0) * 100

    # Plot win rate
    ax.text(
        0.13,
        0.85,
        rf"$w_e={win_rate:.0f}\%$",
        fontsize=fontsize,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=compound_colors[element],
        fontweight="bold",
    )

    # Plot bars
    ax.bar(
        energy_points,
        abs(avg_residues),
        color=np.where(avg_residues > 0, "green", "red"),
        width=0.25,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add element label
    ax.text(
        0.04,
        0.85,
        element,
        fontsize=fontsize,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            facecolor=compound_colors[element],
            alpha=0.4,
            edgecolor="white",
        ),
        fontweight="bold",
    )

    # Add VASP label if applicable
    if spectrum_type == SpectrumType.VASP:
        ax.text(
            0.03,
            0.6,
            "VASP",
            fontsize=fontsize * 0.5,
            color=compound_colors[element],
            transform=ax.transAxes,
            va="top",
            ha="left",
        )

    # Set axis limits and ticks
    ax.set_xlim(-0.25, max(energy_points) + 0.25)
    ax.set_ylim(0, None)
    y_max = ax.get_ylim()[1]
    y_tick = round((y_max / 2) * 4) / 4
    ax.set_yticks([0, y_tick])

    # Create and style twin axis
    ax_twin = ax.twinx()
    ax_twin.patch.set_facecolor("white")
    ax_twin.patch.set_alpha(0.85)

    # Plot mean spectra
    ax_twin.plot(
        energy_points,
        train_avg,
        color="#101010",
        linewidth=0.6,
        linestyle="--",
        zorder=3,
    )

    ax_twin.spines["right"].set_color("gray")
    format_twin_ticks(ax_twin, train_avg)

    # Set tick parameters
    ax.tick_params(axis="both", which="major", labelsize=fontsize * 0.8)
    ax.tick_params(axis="x", which="both", top=False)
    ax_twin.tick_params(axis="y", which="major", labelsize=fontsize * 0.8)


def plot_performance_across_energy(
    comparison_metrics: Dict[ModelTag, ComparisonMetrics],
    figsize: tuple = (8, 18),
    fontsize: int = 18,
) -> tuple:
    """
    Create performance comparison plot across energy range.
    """
    plt.style.use(["default", "science"])

    # Setup figure and grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(10, 1, hspace=0.035, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    # Get ordered elements
    feff_elements = list(ElementsFEFF)
    vasp_elements = list(ElementsVASP)
    all_elements = feff_elements + vasp_elements

    # Create color map using tab10
    compound_colors = {
        elem: plt.cm.tab10.colors[i] for i, elem in enumerate(all_elements)
    }

    energy_points = 0.25 * np.arange(141)

    # Plot FEFF data
    for idx, element in enumerate(feff_elements):
        plot_element_panel(
            axs[idx],
            element,
            SpectrumType.FEFF,
            idx,
            compound_colors,
            energy_points,
            comparison_metrics,
            fontsize,
        )

    # Plot VASP data
    for idx, element in enumerate(vasp_elements):
        plot_element_panel(
            axs[-(len(vasp_elements) - idx)],
            element,
            SpectrumType.VASP,
            idx + len(feff_elements),
            compound_colors,
            energy_points,
            comparison_metrics,
            fontsize,
        )

    # Add legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.5),
    ]
    labels = [r"$\tilde{I}(\Delta E) < 0$", r"$\tilde{I}(\Delta E) > 0$"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9225),
        ncol=2,
        frameon=False,
        fontsize=26 * 0.7,
    )

    # Labels
    axs[-1].set_xlabel(r"$\Delta E$ (eV)", fontsize=fontsize)
    fig.text(
        0.035,
        0.5,
        r"$|\tilde{I}(\Delta E)|$",
        va="center",
        rotation="vertical",
        fontsize=fontsize * 1.1,
        ha="center",
    )
    fig.text(
        0.96,
        0.5,
        r"$\mu_{\text{baseline}} \times 10^{3}$",
        va="center",
        rotation=-90,
        fontsize=fontsize * 1.1,
        ha="center",
    )

    fig.tight_layout()
    return fig, axs


# %%

if __name__ == "__main__":
    comparison_metrics = CompareAllExpertAndTuned()
    fig, axs = plot_performance_across_energy(comparison_metrics)
    fig.savefig("performance_across_energy.pdf", dpi=300, bbox_inches="tight")

# %%
