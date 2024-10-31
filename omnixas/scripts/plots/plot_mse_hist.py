# %%
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from omnixas.model.trained_model import MeanModel, ModelMetrics, ModelTag


class MSEHistogramPlotter:
    @staticmethod
    def plot_mse_histograms(
        metrics: Dict[ModelTag, ModelMetrics],
        fig=None,
        axs=None,
    ):
        """Plot MSE histograms for each model/element."""

        ASPECT_RATIO = 4 / 3
        HEIGHT = 10
        WIDTH = HEIGHT / ASPECT_RATIO
        DPI = 300
        FONTSIZE = 22

        plt.style.use(["default", "science"])

        cmap = "tab10"

        n_elements = len(metrics)
        COLS = 2
        ROWS = (n_elements + 1) // COLS  # ceiling division

        if fig is None or axs is None:
            fig, axs = plt.subplots(
                ROWS,
                COLS,
                figsize=(WIDTH, HEIGHT),
                dpi=DPI,
                sharex=True,
                sharey=True,
                gridspec_kw={"hspace": 0.05, "wspace": 0.025},
            )

        color_keys = [f"{tag.element}_{tag.type}" for tag in metrics.keys()]
        colormap = plt.get_cmap(cmap)
        colors = {k: colormap(i) for i, k in enumerate(color_keys)}

        all_log_mses = np.concatenate(
            [np.log10(m.mse_per_spectra) for m in metrics.values()]
        )

        bin_min = np.floor(np.quantile(all_log_mses, 0.01))
        bin_max = np.ceil(np.quantile(all_log_mses, 0.99))
        bins = np.linspace(bin_min, bin_max, 30)

        # Plot each histogram
        for i, (tag, ax) in enumerate(zip(metrics.keys(), axs.flatten())):
            element = tag.element
            color = colors[f"{element}_{tag.type}"]

            # Calculate MSE per spectra
            mse_per_spectra = metrics[tag].mse_per_spectra

            # Plot histogram
            ax.hist(
                np.log10(mse_per_spectra),
                density=True,
                # bins=np.linspace(-4, 0, 30),
                bins=bins,
                color=color,
                edgecolor="black",
                zorder=1,
                alpha=0.6,
            )

            # Add element label
            label_text = f"{element}"
            if tag.type == "VASP":
                label_text = f"{element}"

            ax.text(
                0.95,
                0.92,
                label_text,
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=FONTSIZE * 0.8,
                color="black",
                fontweight="bold",
                bbox=dict(
                    facecolor=color,
                    alpha=0.2,
                    edgecolor=color,
                ),
            )

            # Add VASP label if needed
            if tag.type == "VASP":
                ax.text(
                    0.91,
                    0.66,
                    "VASP",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=FONTSIZE * 0.45,
                    color="black",
                )

            x_min = -10.2
            x_max = -5.8
            # x_min = bin_min - 0.2
            # x_max = bin_max + 0.2

            warnings.warn("Hardcoded x_min and x_max")

            xticks = np.arange(np.ceil(x_min), np.ceil(x_max), 1)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.0f}" for x in xticks], fontsize=FONTSIZE * 0.7)
            ax.set_xlim(x_min, x_max)

            warnings.warn("Hardcoded yticks")
            yticks = [0.3, 0.6, 0.9, 1.2]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks, fontsize=FONTSIZE * 0.7)
            ax.set_ylim(None, 1.3)

            ax.tick_params(
                axis="both", which="both", direction="in", top=False, right=False
            )

            ax.axvline(
                np.log10(MeanModel(tag=tag).metrics.mse),
                color="black",
                linestyle=":",
                alpha=0.5,
                label="baseline",
                zorder=0,
            )

        # Clean up empty subplots if any
        for ax in axs.flatten()[len(metrics) :]:
            ax.set_visible(False)

        # Add labels
        axs[2, 0].set_ylabel("Density", fontsize=FONTSIZE)
        for ax in axs[-1, :]:
            ax.set_xlabel(r"$\log_{10}(\text{MSE})$", fontsize=FONTSIZE)

        # Add legend
        axs[-1, 0].legend(fontsize=FONTSIZE * 0.8, loc="center left")

        plt.tight_layout()
        return fig, axs


def main():

    from omnixas.scripts.plots.scripts import (
        AllExpertMetrics,
        AllTunedMetrics,
        AllUniversalMetrics,
    )

    fig, ax = MSEHistogramPlotter.plot_mse_histograms(AllExpertMetrics())
    fig.savefig("expert_mse_histograms.pdf", bbox_inches="tight", dpi=300)

    fig, ax = MSEHistogramPlotter.plot_mse_histograms(AllTunedMetrics())
    fig.savefig("tuned_mse_histograms.pdf", bbox_inches="tight", dpi=300)

    fig, ax = MSEHistogramPlotter.plot_mse_histograms(AllUniversalMetrics())
    fig.savefig("universal_mse_histograms.pdf", bbox_inches="tight", dpi=300)


# %%
if __name__ == "__main__":
    main()

# %%
