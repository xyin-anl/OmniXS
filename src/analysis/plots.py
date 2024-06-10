import re
from src.models.trained_models import MeanModel
from functools import cached_property

import numpy as np
import scienceplots
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.data.vasp_data import VASPData
from src.models.trained_models import TrainedModel


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

    @property
    def hatch_for_models(self):
        return {"FCModel": ".", "LinReg": "", "MeanModel": "o"}

    def plot_top_predictions(
        self,
        top_predictions,
        compound,
        splits=10,
        axs=None,
        fill=True,
        color_background=False,
    ):
        if axs is None:
            fig, axs = plt.subplots(splits, 1, figsize=(8, 20))
            axs = axs.flatten()
        else:
            assert len(axs) == splits, "Number of subplots must match splits"

        cmap = "tab10"
        compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))
        compound_colors = {
            c: compound_colors[i] for i, c in enumerate(cfg.compounds + ["Ti", "Cu"])
        }

        for i, ax in enumerate(axs):
            t = top_predictions[i][0]
            p = top_predictions[i][1]

            if color_background:
                ax.patch.set_facecolor(compound_colors[compound])
                ax.patch.set_alpha(0.05)

            ax.plot(
                t,
                "-",
                color=compound_colors[compound],
                linewidth=1.5,
            )
            ax.plot(
                p,
                # "--",
                linestyle="",
                color=compound_colors[compound],
                linewidth=1.5,
            )

            if fill:
                ax.fill_between(
                    np.arange(len(top_predictions[i][0])),
                    t,
                    p,
                    alpha=0.6,
                    label=f"Mean Residue {(t-p).__abs__().mean():.1e}",
                    # color="red",
                    color=compound_colors[compound],
                )

            # ax.set_axis_off()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # # remove axis and ticks other than x-axis in last plot
        # axs[-1].set_axis_on()
        # axs[-1].spines["top"].set_visible(False)
        # axs[-1].spines["right"].set_visible(False)
        # axs[-1].spines["left"].set_visible(False)
        # axs[-1].tick_params(axis="y", which="both", left=False, labelleft=False)

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
        axs[-1].set_xlabel(compound, fontsize=20)
        return self

    def save(self, file_name=None, ext="pdf"):
        def fname(string):
            # helper function to create file name from string
            # remove special characters and replace spaces with underscores
            return re.sub(r"[^\w\s]", "", string).replace(" ", "_")

        file_name = file_name or (fname(self.title) if self.title else "untitled")
        plt.savefig(f"{file_name}.{ext}", bbox_inches="tight", dpi=300)
        return self

    def bar_plot_of_loss(self, model_list, ax=None, compare_with_mean_model=False):
        ax = ax or plt.gca()

        colors = [self.colors_for_compounds[m.compound] for m in model_list]

        if not compare_with_mean_model:
            mse = {m.compound: m.mse for m in model_list}
        else:
            mse = {
                m.compound: MeanModel(DataQuery(m.compound, m.simulation_type)).mse
                / m.mse
                for m in model_list
            }

        model_name = set([m.name for m in model_list])
        assert len(model_name) == 1, "All models must be of same type"
        model_name = model_name.pop()

        # bar plots
        ax.bar(
            mse.keys(),
            mse.values(),
            alpha=1.0,
            color=colors,
            edgecolor="black",
            hatch=self.hatch_for_models[model_name],
            label=f"{model_name}",
            # if not compare_with_mean_model
            # else f"MeanModel/{model_name}",
        )
        # values on top of bars

        # # add labels on top of bars for each compound
        for compound, mse in mse.items():
            ax.text(
                compound,
                mse,
                f"{mse:.1e}" if not compare_with_mean_model else f"{mse:.2f}",
                ha="center",
                va="bottom",
                fontsize=16,
                color=self.colors_for_compounds[compound],
            )

        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=18)
        y_label = "MSE" if not compare_with_mean_model else "MeanModel_MSE/Model_MSE"
        ax.set_ylabel(y_label, fontsize=20)
        ax.legend(fontsize=20, loc="upper left")
        ax.set_yscale("log" if not compare_with_mean_model else "linear")

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

    def plot_peak_loc(self, model, compound, ax=None):
        ax = ax or plt.gca()

        # range of energies for compound
        e_start = cfg.transformations.e_start[compound]
        e_diff = cfg.transformations.e_range_diff
        e_end = e_start + e_diff
        e_map = np.linspace(e_start, e_end, model.data.test.y.shape[1])

        # peak locations in eV for ground truth
        peak_truth_idx = np.argmax(model.data.test.y, axis=1)
        peak_loc_truth = e_map[peak_truth_idx]

        # peak locations in eV for predictions
        peak_pred_idx = np.argmax(model.predictions, axis=1)
        peak_loc_predictions = e_map[peak_pred_idx]

        ax.scatter(
            peak_loc_truth,
            peak_loc_predictions,
            color=Plot().colors_for_compounds[compound],
            marker="o",
            facecolors="none",
            linewidths=1.5,
            edgecolors=Plot().colors_for_compounds[compound],
        )

        # x == y line
        x_lims = np.array([np.min(peak_loc_truth), np.max(peak_loc_truth)])
        y_lims = np.array([np.min(peak_loc_predictions), np.max(peak_loc_predictions)])
        ax.plot(x_lims, y_lims, color="gray", linestyle="--")

        # linear fit
        fit = np.polyfit(peak_loc_truth, peak_loc_predictions, 1)
        fit_fn = np.poly1d(fit)
        r2 = r2_score(peak_loc_truth, peak_loc_predictions)
        ax.plot(
            x_lims,
            fit_fn(x_lims),
            linestyle="-",
            color=Plot().colors_for_compounds[compound],
            # label=f"R2 = {r2:.3f}",
            linewidth=1.5,
        )
        # # add text to fit line
        # ax.text(
        #     0.05,
        #     0.95,
        #     f"y = {fit_fn[1]:.3f}x + {fit_fn[0]:.3f}",
        #     transform=ax.transAxes,
        #     fontsize=14,
        #     verticalalignment="top",
        # )

        ax.set_title(f"{compound}: R2={r2:.3f}", fontsize=18)
        # ax.set_xlabel("Ground Truth", fontsize=14)
        # ax.set_ylabel("Predictions", fontsize=14)
        x0 = min([np.min(peak_loc_truth), np.min(peak_loc_predictions)])
        x1 = max([np.max(peak_loc_truth), np.max(peak_loc_predictions)])
        ax.set_xlim(x0, x1)
        ax.set_ylim(x0, x1)
        ax.set_aspect("equal", "box"),
        ax.legend(fontsize=14)
        return r2


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
