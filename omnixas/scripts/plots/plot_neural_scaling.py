# %%
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

from omnixas.data import DataTag
from omnixas.model.trained_model import ModelTag
from omnixas.scripts.plots.scripts import AllMlSplits, ExpertEtas


class PlotNeuralScaling:
    @staticmethod
    def plot_performance_vs_size(
        etas: Dict[str, float],
        data_sizes: Dict[DataTag, int],
    ):
        plt.style.use(["default", "science"])
        FONTSIZE = 20

        fig = plt.figure(figsize=(8, 7), dpi=100)  # Explicitly set dpi
        ax = fig.add_subplot(111)

        # Calculate limits from data
        x_max = max(size for _, size in data_sizes.items())
        y_max = max(eta for eta in etas.values())

        # Add some padding (e.g., 10%)
        x_max = x_max * 1.1
        y_max = y_max * 1.1

        # Round to nice numbers
        x_max = np.ceil(x_max / 1000) * 1000  # Round to nearest thousand
        y_max = np.ceil(y_max / 2) * 2  # Round to nearest 5

        # Set limits
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

        # Create color map ensuring unique colors for each element_type
        colormap = plt.cm.tab10
        all_keys = list(data_sizes.keys())

        colors = {
            f"{tag.element}_{tag.type}": colormap(i) for i, tag in enumerate(all_keys)
        }

        # Plot data points
        for tag, size in data_sizes.items():
            eta = etas[tag]
            color = colors[f"{tag.element}_{tag.type}"]

            ax.scatter(
                size,
                eta,
                s=300,
                # color=color,
                # edgecolor="black",
                edgecolor=color,
                zorder=2,
                marker="s" if tag.type == "FEFF" else "o",
                # alpha=0.5,
                # no fill
                facecolors="none",
                hatch="///////" if tag.type == "VASP" else None,
                label=(
                    f"{tag.element}" if tag.type == "FEFF" else f"{tag.element} VASP"
                ),
            )

            # # Add text label with proper styling
            # ax.text(
            #     size,
            #     eta,
            #     tag.element,
            #     fontsize=FONTSIZE,
            #     ha="center",
            #     va="center",
            #     color="black",
            #     bbox=dict(
            #         boxstyle="square,pad=0.3",
            #         facecolor=color,
            #         alpha=0.4,
            #         edgecolor="black",
            #         hatch="///" if tag.type == "VASP" else None,
            #     ),
            #     zorder=3,
            # )

        # Style the plot
        ax.set_xlabel("Data size", fontsize=FONTSIZE)
        ax.set_ylabel(r"Performance ($\eta$)", fontsize=FONTSIZE)

        xticks = np.arange(
            ax.get_xlim()[0], ax.get_xlim()[1] + 2000, 3000
        )  # Steps of 2000
        warnings.warn("Hardcoded xticks")
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(x):,.0f}" for x in xticks], fontsize=0.8 * FONTSIZE)
        ax.set_yticks(np.arange(4, y_max, 4))
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x))
        )
        ax.set_yticklabels(ax.get_yticks(), fontsize=0.8 * FONTSIZE)

        # Grid and ticks
        ax.minorticks_off()
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        # single row
        ax.legend(
            fontsize=FONTSIZE * 0.8,
            loc="best",
            title="Element",
            # location outside of figure on right
            bbox_to_anchor=(1, 1),
        )

        # ax.legend(
        #     handles=legend_elements,
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 1.15),
        #     ncol=len(legend_elements),  # Single row
        #     fontsize=FONTSIZE * 0.8,
        #     handletextpad=1,
        #     title="Element",
        #     frameon=False
        # )

        # # Custom legend
        # legend_elements = [
        #     Rectangle(
        #         (0, 0), 1, 1, facecolor="lightgray", edgecolor="gray", label="FEFF"
        #     ),
        #     Circle(
        #         (0, 0),
        #         1,
        #         facecolor="white",
        #         edgecolor="gray",
        #         label="VASP",
        #         hatch="///",
        #     ),
        # ]

        # class HandlerSquare(HandlerPatch):
        #     def create_artists(
        #         self,
        #         legend,
        #         orig_handle,
        #         xdescent,
        #         ydescent,
        #         width,
        #         height,
        #         fontsize,
        #         trans,
        #     ):
        #         center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        #         p = mpatches.Rectangle(
        #             xy=(center[0] - width / 2, center[1] - height / 2),
        #             width=width,
        #             height=height,
        #         )
        #         self.update_prop(p, orig_handle, legend)
        #         p.set_transform(trans)
        #         return [p]
        # Add legend
        # ax.legend(
        #     handles=legend_elements,
        #     loc="upper left",
        #     # handler_map={mpatches.Rectangle: HandlerSquare()},
        #     handlelength=3,
        #     handleheight=3,
        # )
        # plt.setp(ax.get_legend().get_texts(), fontsize=FONTSIZE * 0.8)
        # fig.tight_layout()
        # fig.savefig("performance_vs_size.pdf", dpi=300, bbox_inches="tight")

        return fig, ax


def main():
    etas = ExpertEtas()  # pass kwargs here
    splits = AllMlSplits()  # pass kwargs here
    data_sizes = {
        ModelTag(name="expertXAS", **data_tag.dict()): len(split.train)
        for data_tag, split in splits.items()
    }
    fig, ax = PlotNeuralScaling.plot_performance_vs_size(etas, data_sizes)
    fig.show()
    # fig.savefig("performance_vs_size.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()