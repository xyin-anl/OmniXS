import matplotlib.pyplot as plt
import numpy as np
import torch

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import Trained_FCModel
import scienceplots


def plot_top_pred_comparisions(
    model1_name,
    model2_name,
    sim_type="FEFF",
    compounds=cfg.compounds,
    splits=10,
    fontsize=30,
    axs=None,
):
    plt.style.use(["default", "science"])
    if axs is None:
        fig, axs = plt.subplots(
            splits,
            len(compounds),
            figsize=(len(compounds) * 4, 2 * splits),
        )

    for ax_col, compound in zip(axs.T, compounds):
        data = load_xas_ml_data(DataQuery(compound, sim_type))

        model1 = Trained_FCModel(DataQuery(compound, sim_type), name=model1_name)
        model2 = Trained_FCModel(DataQuery(compound, sim_type), name=model2_name)

        mode1_pred_sort_idxs = model1.sorted_predictions()[1]
        plot_idxs = mode1_pred_sort_idxs[:: len(mode1_pred_sort_idxs) // splits]
        for ax, idx in zip(ax_col, plot_idxs):

            gnd_truth = data.test.y[idx]
            model1_pred = model1.model(torch.tensor(data.test.X[idx]).unsqueeze(0))
            model2_pred = model2.model(torch.tensor(data.test.X[idx]).unsqueeze(0))

            model1_pred = model1_pred.detach().numpy().squeeze()
            model2_pred = model2_pred.detach().numpy().squeeze()

            ax.plot(gnd_truth, color="green", label="Ground Truth")
            ax.plot(model1_pred, color="blue", label=model1_name)
            ax.plot(model2_pred, color="red", label=model2_name)

            ax.fill_between(
                np.arange(len(gnd_truth)),
                model1_pred,
                gnd_truth,
                where=abs(model1_pred - gnd_truth) > abs(model2_pred - gnd_truth),
                color="skyblue",
                alpha=0.5,
            )
            ax.fill_between(
                np.arange(len(gnd_truth)),
                model2_pred,
                gnd_truth,
                where=abs(model2_pred - gnd_truth) > abs(model1_pred - gnd_truth),
                color="red",
                alpha=0.5,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])

    for c, ax in zip(compounds, axs[0, :]):
        ax.set_title(c, fontsize=fontsize)
    suptitle = f"Top 5 predictions for {model1_name} and {model2_name} models"
    suptitle += "\n Fill color is based on color of worst model"
    fig = axs[0, 0].get_figure()
    fig.suptitle(suptitle, fontsize=fontsize, y=1.01)
    fig.legend(["Ground Truth", model1_name, model2_name], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"improvement_regions_{sim_type}.pdf", bbox_inches="tight", dpi=400)


if __name__ == "__main__":

    splits = 10
    model1_name = "per_compound_tl"
    model2_name = "ft_tl"

    # # =============================================================================
    # # FEFF
    # # =============================================================================
    # feff_compounds = cfg.compounds
    # plot_top_pred_comparisions(
    #     model1_name=model1_name,
    #     model2_name=model2_name,
    #     sim_type="FEFF",
    #     compounds=feff_compounds,
    # )
    # plt.show()
    # # =============================================================================

    # =============================================================================
    # VASP
    # =============================================================================
    vasp_compounds = ["Cu", "Ti"]
    fig, axs = plt.subplots(
        splits, len(vasp_compounds), figsize=(len(vasp_compounds) * 4, 2 * splits)
    )
    plot_top_pred_comparisions(
        model1_name=model1_name,
        model2_name=model2_name,
        sim_type="VASP",
        compounds=vasp_compounds,
        fontsize=10,
        axs=axs,
    )
    plt.show()
    # =============================================================================
