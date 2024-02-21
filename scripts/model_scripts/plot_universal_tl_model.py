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


def univ_mse_per_compound():
    data_all, compound_labels = load_all_data(return_compound_name=True)
    universal_model = Trained_FCModel(DataQuery("ALL", "FEFF"))
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
        universal_mse_per_compound[c] = universal_model.mse
    return universal_mse_per_compound


# bar plot for fc_mse and mse_universal
def plot_performace_of_universal_tl_model(
    fc_mse,
    universal_mse_per_compound,
    universal_mse,
):
    plt.style.use(["default", "science"])
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.25
    compounds = cfg.compounds
    n_groups = len(compounds)
    index = np.arange(n_groups)
    values = [fc_mse[c] for c in compounds]
    ax.bar(index, values, bar_width, label="Per-compound-TL-MLP", edgecolor="black")
    values = [universal_mse_per_compound[c] for c in compounds]
    ax.bar(
        index + bar_width,
        values,
        bar_width,
        label="Universal-TL-MLP",
        edgecolor="black",
    )
    ax.set_xlabel("Compound", fontsize=20)
    ax.set_ylabel("MSE", fontsize=20)
    ax.set_title("Per-compound-TL-MLP vs Universal-TL-MLP", fontsize=24)
    ax.axhline(universal_mse, color="red", linestyle="--", label="Universal_TL MSE")
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(compounds, fontsize=18)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig("per_compound_vs_universal_tl_mlp.pdf", bbox_inches="tight", dpi=300)


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


if __name__ == "__main__":
    universal_mse_per_compound = univ_mse_per_compound()
    universal_mse = Trained_FCModel(DataQuery("ALL", "FEFF")).mse
    fc_mse = {c: Trained_FCModel(DataQuery(c, "FEFF")).mse for c in cfg.compounds}
    plot_performace_of_universal_tl_model(
        fc_mse, universal_mse_per_compound, universal_mse
    )
    plot_deciles_of_universal_tl_model()
