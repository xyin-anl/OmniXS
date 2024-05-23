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

    # plt.suptitle(
    #     f"{simulation_type}: Top {splits} predictions for {model_name}",
    #     fontsize=24,
    # )

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
    add_weighted=False,
    FONTSIZE=18,
):
    plt.style.use(["default", "science"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.2
    n_groups = len(cfg.compounds) if not include_vasp else len(cfg.compounds) + 2
    index = np.arange(n_groups)

    tableau_colorblind10 = [
        "#006BA4",
        "#FF800E",
        "#ABABAB",
        "#595959",
        "#5F9ED1",
        "#C85200",
        "#898989",
        "#A2C8EC",
        "#FFBC79",
        "#CFCFCF",
    ]

    colors = {
        "univ_TL_MLP": tableau_colorblind10[0],
        "per_compound_tl": tableau_colorblind10[1],
        "ft_tl": tableau_colorblind10[2],
    }

    sims = list(zip(cfg.compounds, ["FEFF"] * len(cfg.compounds)))
    if include_vasp:
        sims += [("Ti", "VASP"), ("Cu", "VASP")]

    univ_mses = universal_TL_mses(relative_to_per_compound_mean_model)
    ax.bar(
        # index,
        np.arange(len(cfg.compounds)),  # VASP has no universal
        univ_mses["per_compound"].values(),
        bar_width,
        label="Universal-TL-MLP",
        edgecolor=None,
        color=colors["univ_TL_MLP"],
        zorder=3,
    )

    for i, model_name in enumerate(model_names, start=1):
        fc_models = [
            Trained_FCModel(DataQuery(c, sim_type), name=model_name)
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
            # edgecolor="black",
            edgecolor=None if i < 8 else colors[model_name],
            fill=True if i < 8 else None,
            # hatch=(
            #     [""] * len(cfg.compounds) + ["....."] * 2
            #     if include_vasp
            #     else [""] * len(cfg.compounds)
            # ),
            zorder=3,
        )

    if add_weighted:
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
    y_label = (
        "MSE mean model / MSE model" if relative_to_per_compound_mean_model else "MSE"
    )

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

    ax.set_xlabel("Compound", fontsize=FONTSIZE * 1.2, labelpad=-10)
    ax.set_ylabel(y_label, fontsize=FONTSIZE * 1.2)
    # ax.set_title(title, fontsize=24)
    ax.set_xticks(index + bar_width)

    ax.set_xticklabels(
        cfg.compounds
        + (
            [
                "Ti\n" + r"{\large VASP}",
                "Cu\n" + r"{\large VASP}",
            ]
            if include_vasp
            else []
        ),
        fontsize=FONTSIZE,
    )

    ax.legend(fontsize=FONTSIZE)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.savefig(file_name[:-4] + ".png", bbox_inches="tight", dpi=300)
    plt.show()
    return ax


if __name__ == "__main__":

    # plot_universal_tl_vs_per_compound_tl(relative_to_per_compound_mean_model=True)

    import matplotlib as mpl

    ASPECT_RATIO = 4 / 3
    HEIGHT = 6
    WEIGHT = HEIGHT * ASPECT_RATIO
    DPI = 300
    FONTSIZE = 14
    plt.style.use(["default", "science", "tableau-colorblind10"])

    # mpl.rcParams["font.size"] = FONTSIZE
    # mpl.rcParams["axes.labelsize"] = FONTSIZE
    # mpl.rcParams["xtick.labelsize"] = FONTSIZE
    # mpl.rcParams["ytick.labelsize"] = FONTSIZE
    # mpl.rcParams["legend.fontsize"] = FONTSIZE
    # mpl.rcParams["figure.dpi"] = DPI
    # mpl.rcParams["figure.figsize"] = (WEIGHT, HEIGHT)
    # mpl.rcParams["savefig.dpi"] = DPI
    # mpl.rcParams["savefig.format"] = "pdf"
    # mpl.rcParams["savefig.bbox"] = "tight"

    fig, ax = plt.subplots(1, 1, figsize=(WEIGHT, HEIGHT), dpi=DPI)

    plot_universal_tl_vs_per_compound_tl(
        relative_to_per_compound_mean_model=True,
        include_vasp=True,
        ax=ax,
        add_weighted=False,
        FONTSIZE=FONTSIZE,
    )

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

    # plot_deciles_of_top_predictions(
    #     model_name="ft_tl",
    #     fixed_model=None,
    #     compounds=["Ti", "Cu"],
    #     simulation_type="VASP",
    # )
