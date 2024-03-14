# =============================================================================
# Plots related to Universal-TL-MLP model
# =============================================================================

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
    universal_model = Trained_FCModel(DataQuery("ALL", "FEFF"))

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


def plot_deciles_of_universal_tl_model():
    splits = 10
    universal_top_predictions = {}
    universal_model = Trained_FCModel(DataQuery("ALL", "FEFF"))
    for c in cfg.compounds:
        universal_model.data = load_xas_ml_data(DataQuery(c, "FEFF"))
        universal_top_predictions[c] = universal_model.top_predictions(splits=splits)
    fig, axs = plt.subplots(splits, len(cfg.compounds), figsize=(20, 16))
    for i, compound in enumerate(cfg.compounds):
        Plot().set_title(f"{compound}").plot_top_predictions(
            universal_top_predictions[compound],
            splits=splits,
            axs=axs[:, i],
            compound=compound,
        )
    plt.suptitle(
        f"Top {splits} predictions for Universal-TL-MLP for each compound", fontsize=24
    )
    plt.tight_layout()
    plt.savefig(
        "universal_tl_top_predictions_per_compound.pdf", bbox_inches="tight", dpi=300
    )


def plot_universal_tl_vs_per_compound_tl(relative_to_per_compound_mean_model=False):
    plt.style.use(["default", "science"])
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.25
    n_groups = len(cfg.compounds)
    index = np.arange(n_groups)

    colors = {
        "univ_TL_MLP": "red",
        "per_compound_TL_MLP": "blue",
    }

    fc_models = [Trained_FCModel(DataQuery(c, "FEFF")) for c in cfg.compounds]
    fc_mses = [
        (
            model.mse_relative_to_mean_model
            if relative_to_per_compound_mean_model
            else model.mse
        )
        for model in fc_models
    ]
    ax.bar(
        index,
        fc_mses,
        bar_width,
        color=colors["per_compound_TL_MLP"],
        label="Per-compound-TL-MLP",
        edgecolor="black",
    )

    univ_mses = universal_TL_mses(relative_to_per_compound_mean_model)
    ax.bar(
        index + bar_width,
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
        data_sizes = [len(model.data.test.y) for model in fc_models]
        fc_mse_weighted_mse = sum(
            [model.mse * size for model, size in zip(fc_models, data_sizes)]
        ) / sum(data_sizes)
        ax.axhline(
            fc_mse_weighted_mse,
            color=colors["per_compound_TL_MLP"],
            linestyle="--",
            label="Per_compound_TL_weighted_MSE",
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
    ) + ".pdf"

    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(cfg.compounds, fontsize=18)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    from src.models.trained_models import MeanModel

    plot_universal_tl_vs_per_compound_tl(relative_to_per_compound_mean_model=True)
    plot_universal_tl_vs_per_compound_tl(relative_to_per_compound_mean_model=False)
