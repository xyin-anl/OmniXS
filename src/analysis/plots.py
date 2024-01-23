import scienceplots
from config.defaults import cfg
from src.models.trained_models import TrainedModel
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.data.vasp_data import VASPData

from functools import cached_property
from matplotlib import pyplot as plt
import numpy as np
import re


class Plot:
    def __init__(self):
        self.style()
        self.title = None

    @cached_property
    def colors_for_compounds(self):
        compounds = cfg.compounds
        colormap = plt.cm.get_cmap("tab10")
        color_idx = np.linspace(0, 1, len(compounds))
        compound_colors = {c: colormap(color_idx[i]) for i, c in enumerate(compounds)}
        return compound_colors

    def style(self):
        plt.style.use(["default", "science"])
        return self

    def reset(self):
        plt.style.use("default")
        self.title = None
        return self

    def set_title(self, title):
        self.title = title
        return self

    def plot_top_predictions(self, top_predictions, splits=10, axs=None, fill=True):
        if axs is None:
            fig, axs = plt.subplots(splits, 1, figsize=(8, 20))
            axs = axs.flatten()
        else:
            assert len(axs) == splits, "Number of subplots must match splits"

        for i, ax in enumerate(axs):
            t = top_predictions[i][0]
            p = top_predictions[i][1]

            ax.plot(t, "-", color="black", linewidth=1.5)
            ax.plot(p, "--", color="red", linewidth=1.5)

            if fill:
                ax.fill_between(
                    np.arange(len(top_predictions[i][0])),
                    t,
                    p,
                    alpha=0.4,
                    label=f"Mean Residue {(t-p).__abs__().mean():.1e}",
                    color="red",
                )

            ax.set_axis_off()

        # remove axis and ticks other than x-axis in last plot
        axs[-1].set_axis_on()
        axs[-1].spines["top"].set_visible(False)
        axs[-1].spines["right"].set_visible(False)
        axs[-1].spines["left"].set_visible(False)
        axs[-1].tick_params(axis="y", which="both", left=False, labelleft=False)

        # set x-axis ticks labels based on cfg values
        e_start = cfg.transformations.e_start[compound]
        e_diff = cfg.transformations.e_range_diff
        e_end = e_start + e_diff
        axs[-1].set_xticks(np.linspace(0, len(top_predictions[0][0]), 2))
        axs[-1].set_xticklabels(
            [
                f"{e:.1f}"
                for e in np.linspace(e_start, e_end, len(axs[-1].get_xticklabels()))
            ],
            fontsize=14,
        )
        axs[-1].set_xlabel(self.title, fontsize=20)
        return self

    def save(self, file_name=None, ext="pdf"):
        def fname(string):
            # helper function to create file name from string
            # remove special characters and replace spaces with underscores
            return re.sub(r"[^\w\s]", "", string).replace(" ", "_")

        file_name = file_name or (fname(self.title) if self.title else "untitled")
        plt.savefig(f"{file_name}.{ext}", bbox_inches="tight", dpi=300)
        return self

    def bar_plot_of_loss(self, model_list, ax=None):
        ax = ax or plt.gca()

        mse = {m.compound: m.mse for m in model_list}
        model_name = set([m.name for m in model_list])
        assert len(model_name) == 1, "All models must be of same type"
        model_name = model_name.pop()

        # bar plots
        ax.bar(mse.keys(), mse.values(), label=model_name, alpha=0.5)
        # values on top of bars
        for i, (c, m) in enumerate(mse.items()):
            ax.text(i, m, f"{m:.1e}", ha="center", va="bottom", fontsize=12)

        ax.set_ylabel("MSE")
        ax.set_xlabel("Compound")
        ax.legend()
        ax.set_yscale("log")

        # make sure the text are not outside the plot
        return self

    # def histogram_of_residues(model, ax=None):
    #     residues = model.residues()
    #     compound = model.compound
    #     ax = ax or plt.gca()
    #     # binwidth = 0.000001
    #     # bins = np.arange(min(residues), max(residues) + binwidth, binwidth)
    #     # counts, bin_edges = np.histogram(np.log(residues), bins=bins)
    #     # counts, bin_edges = np.histogram(np.log10(residues), bins=10)
    #     # counts = counts.astype(float) / counts.sum()
    #     # ax.step(bin_edges[:-1], counts, where="post", label=f"{compound}")
    #     ax.hist(np.log10(residues), bins=20, label=f"{compound}", alpha=0.8)
    #     ax.legend(fontsize=20)
    #     ax.set_yscale("log")
    #     # ax.set_xscale("log")
    #     # ax.set_ylim(1e-3, 1)
    #     # ax.set_xlim(np.quantile(residues, 0.02), np.quantile(residues, 0.70))


if __name__ == "__main__":
    from src.models.trained_models import LinReg

    splits = 10
    top_predictions = {
        c: LinReg(c).top_predictions(splits=splits) for c in cfg.compounds
    }

    fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(20, 16))
    for i, compound in enumerate(cfg.compounds):
        Plot().set_title(f"{compound}").plot_top_predictions(
            top_predictions[compound], splits=splits, axs=axs[:, i]
        )
    plt.tight_layout(pad=0)
    plt.savefig(f"top_{splits}_predictions.pdf", bbox_inches="tight", dpi=300)
