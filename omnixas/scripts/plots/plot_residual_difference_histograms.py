# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from omnixas.data import ElementsFEFF, ElementsVASP


def plot_residual_difference_histograms(residual_diffs, figsize=(9, 12), fontsize=22):
    plt.style.use(["default", "science"])
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(5, 2, hspace=0.05, wspace=0.02)
    axs = gs.subplots(sharex=True, sharey=True)

    # Get ordered elements
    feff_elements = list(ElementsFEFF)
    vasp_elements = list(ElementsVASP)

    # Create iterator for elements and simulation types
    iterator = [(elem, "FEFF") for elem in feff_elements] + [
        (elem, "VASP") for elem in vasp_elements
    ]

    for idx, ((element, sim_type), ax) in enumerate(zip(iterator, axs.flatten())):
        key = f"{element}_{sim_type}"
        if key not in residual_diffs:
            continue

        # Get and process data
        data = residual_diffs[key].flatten()
        good = data[data > 0]
        bad = data[data < 0]
        good = np.log10(good)
        bad = np.log10(-bad)

        # Histogram parameters
        # bins = np.linspace(-5, 0, 40) # ArXiv Version

        bins = np.linspace(-7.5, -3.2, 40)

        count = len(data)

        # Plot positive differences
        bars = np.histogram(good, bins=bins)[0]
        bars = (bars / count) * 100
        ax.bar(
            bins[:-1],
            bars,
            width=np.diff(bins),
            color=plt.get_cmap("tab10")(idx),
            label=r"$\Delta > 0$",
            edgecolor="black",
            linewidth=0.5,
        )

        # Plot negative differences
        bars = np.histogram(bad, bins=bins)[0]
        bars = (bars / count) * 100
        ax.bar(
            bins[:-1],
            bars,
            width=np.diff(bins),
            alpha=0.7,
            label=r"$\Delta < 0$",
            edgecolor="black",
            linewidth=0.5,
            color="white",
        )

        # Styling
        # ax.set_xticks(np.arange(-4, 0, 1))
        # get tiv valucks as int withn the bar range
        ax.set_xticks(np.arange(-7, -3, 1))
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize * 0.8)
        ax.set_yticks(np.arange(1, 7, 2))
        ax.set_yticklabels(
            [f"{y:.0f}%" for y in ax.get_yticks()], fontsize=fontsize * 0.8
        )

        # Element label
        ax.text(
            0.95,
            0.9,
            element,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            bbox=dict(
                facecolor=plt.get_cmap("tab10")(idx),
                alpha=0.5,
                edgecolor=plt.get_cmap("tab10")(idx),
            ),
        )

        # VASP label if applicable
        if sim_type == "VASP":
            ax.text(
                0.97,
                0.65,
                "VASP",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=fontsize * 0.6,
            )

        # Tick parameters
        ax.tick_params(axis="x", which="both", top=False, direction="out")
        ax.tick_params(axis="y", which="both", right=False, direction="in")
        ax.minorticks_off()

    handles = [
        Patch(facecolor="black", edgecolor="black", label=r"$\Delta>0$"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.7, label=r"$\Delta<0$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        fontsize=fontsize,
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),
        bbox_transform=fig.transFigure,
    )

    # Labels
    xlabel = r"$\log_{10}( |\Delta| )$"
    axs[-1, 0].set_xlabel(xlabel, fontsize=fontsize)
    axs[-1, 1].set_xlabel(xlabel, fontsize=fontsize)

    fig.text(
        0.06,
        0.5,
        r"Frequency Distribution (\%)",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    fig.tight_layout()
    return fig, axs


# %%
if __name__ == "__main__":

    from omnixas.data import ThousandScaler
    from omnixas.scripts.plots.scripts import CompareAllExpertAndTuned
    from omnixas.utils import DEFAULTFILEHANDLER

    residual_diffs = {
        f"{t.element}_{t.type}": metric_diff.residual_diff
        for t, metric_diff in CompareAllExpertAndTuned(
            file_handler=DEFAULTFILEHANDLER(),
            x_scaler=ThousandScaler,
            y_scaler=ThousandScaler,
        ).items()
    }

    fig, axs = plot_residual_difference_histograms(residual_diffs)
    plt.savefig("residual_difference_histograms.pdf", bbox_inches="tight", dpi=300)

# %%
