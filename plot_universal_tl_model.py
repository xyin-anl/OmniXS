# %%
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
from src.models.trained_models import LinReg


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

    fig, axs = plt.subplots(
        splits,
        len(compounds),
        figsize=(20, 16),
        sharex=True,
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    for c, axs in zip(compounds, axs.T):
        Plot().set_title(f"{c}").plot_top_predictions(
            top_predictions[c],
            splits=splits,
            axs=axs,
            compound=c,
            color_background=True,
        )

    # for i, compound in enumerate(compounds):
    #     Plot().set_title(f"{compound}").plot_top_predictions(
    #         top_predictions[compound],
    #         splits=splits,
    #         axs=axs[:, i],
    #         compound=compound,
    #     )

    # ax.patch.set_facecolor(compounds_colors[compound])
    # ax.patch.set_alpha(0.2)

    # plot the background color

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
    model_names=None,  # ["per_compound_tl", "ft_tl"],
    relative_to_per_compound_mean_model=False,
    include_vasp: bool = False,
    ax=None,
    add_weighted=False,
    FONTSIZE=18,
    plot_based_on_rmse=False,
    include_linreg=False,
):
    plt.style.use(["default", "science"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.2
    n_groups = len(cfg.compounds) if not include_vasp else len(cfg.compounds) + 2
    index = np.arange(n_groups)

    # compound_colors = [ # Tableau colorblind 10
    #     "#006BA4",
    #     "#FF800E",
    #     "#ABABAB",
    #     "#595959",
    #     "#5F9ED1",
    #     "#C85200",
    #     "#898989",
    #     "#A2C8EC",
    #     "#FFBC79",
    #     "#CFCFCF",
    # ]

    cmap = "tab10"
    compound_colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(cfg.compounds) + 2))

    colors = {
        "universal_feff": compound_colors[0],
        "per_compound_tl": compound_colors[1],
        "ft_tl": compound_colors[2],
        "linreg": compound_colors[3],
    }

    hatches = {
        "universal_feff": "..",
        "per_compound_tl": ".....",
        "ft_tl": "",
        "linreg": "////",
    }

    sims = list(zip(cfg.compounds, ["FEFF"] * len(cfg.compounds)))
    if include_vasp:
        sims += [("Ti", "VASP"), ("Cu", "VASP")]

    univ_mses = universal_TL_mses(relative_to_per_compound_mean_model)

    if plot_based_on_rmse:
        if relative_to_per_compound_mean_model:
            univ_mses["global"] = np.sqrt(univ_mses["global"])
            univ_mses["per_compound"] = {
                c: np.sqrt(mse) for c, mse in univ_mses["per_compound"].items()
            }
        else:
            univ_mses["global"] = np.sqrt(univ_mses["global"] / 1000**2)
            univ_mses["per_compound"] = {
                c: np.sqrt(mse / 1000**2)
                for c, mse in univ_mses["per_compound"].items()
            }

    model_names = model_names or ["universal_feff", "per_compound_tl", "ft_tl"]
    model_names = ["linreg"] + model_names if include_linreg else model_names

    _bar_loc_dict = {
        "universal_feff": 0,
        "per_compound_tl": 1,
        "ft_tl": 2,
    }

    if include_linreg:  # put linreg at the beginning
        for k, v in _bar_loc_dict.items():
            _bar_loc_dict[k] = v + 1
        _bar_loc_dict["linreg"] = 0

    ax.bar(
        # index,
        np.arange(len(cfg.compounds)) + _bar_loc_dict["universal_feff"] * bar_width,
        univ_mses["per_compound"].values(),
        bar_width,
        label="universal-feff-tl",
        edgecolor=compound_colors,
        fill=False,
        color=compound_colors,
        zorder=3,
        hatch=hatches["universal_feff"],
    )

    _model_dict = {
        "per_compound_tl": Trained_FCModel,
        "ft_tl": Trained_FCModel,
        "linreg": LinReg,
    }
    for model_name in model_names:

        if model_name == "universal_feff":
            continue

        fc_models = [
            _model_dict[model_name](DataQuery(c, sim_type), name=model_name)
            for c, sim_type in sims
        ]
        fc_residues = [
            (
                model.mse_relative_to_mean_model
                if relative_to_per_compound_mean_model
                else model.mse
            )
            for model in fc_models
        ]

        if plot_based_on_rmse:
            if relative_to_per_compound_mean_model:
                fc_residues = [np.sqrt(mse) for mse in fc_residues]
            else:
                fc_residues = [
                    np.sqrt(mse / 1000**2) for mse in fc_residues
                ]  # coz data was scaled by 1000 previously

        bar_positions = np.array(index) + _bar_loc_dict[model_name] * bar_width

        if include_linreg and model_name == "linreg":
            bar_positions[-2:] = bar_positions[-2:] + bar_width

        ax.bar(
            bar_positions,
            fc_residues,
            bar_width,
            color=compound_colors,
            label=model_name,
            fill=False if model_name != "ft_tl" else True,
            hatch=hatches[model_name],
            edgecolor=compound_colors,
            zorder=3,
        )

    if add_weighted:
        ax.axhline(
            univ_mses["global"],
            color=colors["universal_feff"],
            linestyle="--",
            label="Universal_FEFF_global_MSE",
        )

    if (
        not relative_to_per_compound_mean_model and add_weighted
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

    if plot_based_on_rmse:
        if relative_to_per_compound_mean_model:
            y_label = "RMSE mean model / RMSE model"
        else:
            y_label = "RMSE"
    else:
        if relative_to_per_compound_mean_model:
            y_label = "MSE mean model / MSE model"
        else:
            y_label = "MSE"
    ax.set_ylabel(y_label, fontsize=FONTSIZE * 1.2)

    file_name = (
        "per_compound_tl_vs_universal_tl_mlp"
        if not relative_to_per_compound_mean_model
        else "per_compound_tl_vs_universal_tl_relative"
    ) + f"_{len(model_names)}.pdf"

    ax.set_xlabel("Compound", fontsize=FONTSIZE * 1.2, labelpad=-10)
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

    # add legends with hatches in gray
    def hatch_fn(hatch):
        return plt.Rectangle((0, 0), 1, 1, fc="w", edgecolor="black", hatch=hatch)

    ax.legend(
        [hatch_fn(hatches[model_name]) for model_name in model_names],
        model_names,
        fontsize=FONTSIZE * 0.8,
    )

    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.savefig(file_name[:-4] + ".png", bbox_inches="tight", dpi=300)
    plt.show()
    return ax


# %%

# if __name__ == "__main__":

# plot_universal_tl_vs_per_compound_tl(relative_to_per_compound_mean_model=True)

import matplotlib as mpl

ASPECT_RATIO = 4 / 3
HEIGHT = 6
WEIGHT = HEIGHT * ASPECT_RATIO
DPI = 300
FONTSIZE = 14
plt.style.use(["default", "science", "tableau-colorblind10"])
mpl.rcParams["font.size"] = FONTSIZE
mpl.rcParams["axes.labelsize"] = FONTSIZE
mpl.rcParams["xtick.labelsize"] = FONTSIZE
mpl.rcParams["ytick.labelsize"] = FONTSIZE
mpl.rcParams["legend.fontsize"] = FONTSIZE
mpl.rcParams["figure.dpi"] = DPI
mpl.rcParams["figure.figsize"] = (WEIGHT, HEIGHT)
mpl.rcParams["savefig.dpi"] = DPI
mpl.rcParams["savefig.format"] = "pdf"
mpl.rcParams["savefig.bbox"] = "tight"

fig, ax = plt.subplots(1, 1, figsize=(WEIGHT, HEIGHT), dpi=DPI)
plot_universal_tl_vs_per_compound_tl(
    relative_to_per_compound_mean_model=True,
    include_vasp=True,
    ax=ax,
    add_weighted=False,
    FONTSIZE=FONTSIZE,
    plot_based_on_rmse=True,
    include_linreg=True,
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

# %%
