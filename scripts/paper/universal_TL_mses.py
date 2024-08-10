# %%
from typing import Literal

import numpy as np

from config.defaults import cfg
from src.data.ml_data import DataQuery, DataSplit, MLSplits, load_all_data
from src.models.trained_models import MeanModel, Trained_FCModel


def compute_universal_model_metrics(
    metric: Literal[
        "mse",
        "geometric_mean_of_mse_per_spectra",
    ] = "mse",
    compute_relative: bool = True,
):

    data_all, compound_labels = load_all_data(return_compound_name=True)
    universal_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")

    global_metric = getattr(universal_model, metric)

    metric_per_compound = {}
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

        metric_per_compound[c] = getattr(universal_model, metric)
        if compute_relative:
            metric_per_compound[c] = (
                getattr(MeanModel(DataQuery(c, "FEFF")), metric)
                / metric_per_compound[c]
            )

    return {"global": global_metric, "per_compound": metric_per_compound}


# %%

compute_universal_model_metrics()

# %%
