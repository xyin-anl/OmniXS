# =============================================================================
# Plots related to Universal-TL-MLP model
# =============================================================================

from src.models.trained_models import MeanModel
from src.data.ml_data import load_all_data
from src.data.ml_data import MLSplits, DataSplit
from src.models.trained_models import Trained_FCModel
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.analysis.plots import Plot
from config.defaults import cfg
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property


def universal_TL_mses(
    relative_to_per_compound_mean_model=False,
):  # returns {"global": .., "per_compound": {"c": ...}}

    data_all, compound_labels = load_all_data(return_compound_name=True)
    universal_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")

    global_mse = (
        universal_model.mse_relative_to_mean_model
        if relative_to_per_compound_mean_model
        else universal_model.mse
    )

    universal_mse_per_compound = {}
    for c in cfg.compounds:
        splits = []
        for labels, split in zip(
            compound_labels,
            [data_all.train, data_all.val, data_all.test],
        ):
            idx = np.where(labels == c)
            X = split.X[idx]
            y = split.y[idx]
            splits.append(DataSplit(X, y))
        ml_splits = MLSplits(*splits)
        universal_model.data = ml_splits

        universal_mse_per_compound[c] = (
            universal_model.mse
            if not relative_to_per_compound_mean_model
            else MeanModel(DataQuery(c, "FEFF")).mse / universal_model.mse
        )

    return {"global": global_mse, "per_compound": universal_mse_per_compound}


def plot_deciles_of_top_predictions(
    model_name="per_compound_tl",
    fixed_model: Trained_FCModel = None,
    compounds=cfg.compounds,
    simulation_type="FEFF",
):
    splits = 10
    top_predictions = {}
    # universal_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
    for c in compounds:
        if fixed_model is not None:
            model = fixed_model
            model.data = load_xas_ml_data(DataQuery(c, "FEFF"))
        else:
            model = Trained_FCModel(DataQuery(c, simulation_type), name=model_name)
        top_predictions[c] = model.top_predictions(splits=splits)

    fig, axs = plt.subplots(splits, len(compounds), figsize=(20, 16))
    for i, compound in enumerate(compounds):
        Plot().set_title(f"{compound}").plot_top_predictions(
            top_predictions[compound],
            splits=splits,
            axs=axs[:, i],
            compound=compound,
        )
    plt.suptitle(
        f"{simulation_type}: Top {splits} predictions for {model_name}",
        fontsize=24,
    )
    plt.tight_layout()
    plt.savefig(
        f"top_predictions_{model_name}_{simulation_type}.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_universal_tl_vs_per_compound_tl(
    model_names=["per_compound_tl", "ft_tl"],
    relative_to_per_compound_mean_model=False,
    include_vasp: bool = False,
    ax=None,
):
    plt.style.use(["default", "science"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.25
    n_groups = len(cfg.compounds) if not include_vasp else len(cfg.compounds) + 2
    index = np.arange(n_groups)

    colors = {
        "univ_TL_MLP": "red",
        "per_compound_tl": "blue",
        "ft_tl": "green",
    }

    sims = list(zip(cfg.compounds, ["FEFF"] * len(cfg.compounds)))
    if include_vasp:
        sims += [("Ti", "VASP"), ("Cu", "VASP")]

    for i, model_name in enumerate(model_names, start=1):
        fc_models = [
            Trained_FCModel(DataQuery(c, sim_type), name=model_name)
            # for c in cfg.compounds
            for c, sim_type in sims
        ]
        fc_mses = [
            (
                model.mse_relative_to_mean_model
                if relative_to_per_compound_mean_model
                else model.mse
            )
            for model in fc_models
        ]
        ax.bar(
            index + i * bar_width,
            fc_mses,
            bar_width,
            color=colors[model_name],
            label=model_name,
            edgecolor="black",
            hatch=(
                [""] * len(cfg.compounds) + ["\\"] * 2
                if include_vasp
                else [""] * len(cfg.compounds)
            ),
        )

    univ_mses = universal_TL_mses(relative_to_per_compound_mean_model)
    ax.bar(
        # index,
        np.arange(len(cfg.compounds)),  # VASP has no universal
        univ_mses["per_compound"].values(),
        bar_width,
        label="Universal-TL-MLP",
        edgecolor="black",
        color=colors["univ_TL_MLP"],
    )
    ax.axhline(
        univ_mses["global"],
        color=colors["univ_TL_MLP"],
        linestyle="--",
        label="Universal_TL_global_MSE",
    )

    if (
        not relative_to_per_compound_mean_model
    ):  # coz weighte mean has no meaning in relative case
        for model_name in model_names:
            fc_models = [
                Trained_FCModel(DataQuery(c, "FEFF"), name=model_name)
                for c in cfg.compounds
            ]
            data_sizes = [len(model.data.test.y) for model in fc_models]
            fc_mse_weighted_mse = sum(
                [model.mse * size for model, size in zip(fc_models, data_sizes)]
            ) / sum(data_sizes)
            ax.axhline(
                fc_mse_weighted_mse,
                color=colors[model_name],
                linestyle="--",
                label=f"{model_name}_weighted_MSE",
            )

    title = "Per-compound-TL-MLP vs Universal-TL-MLP"
    x_label = "Compound"
    y_label = "Relative MSE" if relative_to_per_compound_mean_model else "MSE"
    title += (
        "\n(relative to per-compound-mean-model)"
        if relative_to_per_compound_mean_model
        else ""
    )
    file_name = (
        "per_compound_tl_vs_universal_tl_mlp"
        if not relative_to_per_compound_mean_model
        else "per_compound_tl_vs_universal_tl_relative"
    ) + f"_{len(model_names)}.pdf"

    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.set_xticks(index + bar_width)
    xlabels = (
        cfg.compounds + ["Ti\nVASP", "Cu\nVASP"] if include_vasp else cfg.compounds
    )
    ax.set_xticklabels(
        xlabels,
        fontsize=18,
    )
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    # plt.show()
    return ax


if __name__ == "__main__":

    # plot_universal_tl_vs_per_compound_tl(relative_to_per_compound_mean_model=True)

    # plot_universal_tl_vs_per_compound_tl(
    #     relative_to_per_compound_mean_model=True,
    #     include_vasp=True,
    # )

    # plot_deciles_of_top_predictions(
    #     model_name="per_compound_tl",
    #     fixed_model=None,
    # )

    # plot_deciles_of_top_predictions(
    #     model_name="ft_tl",
    #     fixed_model=None,
    # )

    # plot_deciles_of_top_predictions(
    #     model_name="universal_tl",
    #     fixed_model=Trained_FCModel(
    #         DataQuery("ALL", "FEFF"), name="universal_tl"
    #     ),  # uses same model for all predictions
    # )

    plot_deciles_of_top_predictions(
        model_name="ft_tl",
        fixed_model=None,
        compounds=["Ti", "Cu"],
        simulation_type="VASP",
    )
