# %%
from functools import partial

import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from omnixas.data import (
    AllDataTags,
    DataTag,
    FEFFDataTags,
    MLSplits,
    ScaledMlSplit,
    ThousandScaler,
    VASPDataTags,
)
from omnixas.model.trained_model import (
    ComparisonMetrics,
    MeanModel,
    ModelTag,
    TrainedModelLoader,
    TrainedXASBlock,
)
from omnixas.utils.io import DEFAULTFILEHANDLER, FileHandler

# %%


def get_eta(model_tag: ModelTag, file_handler: FileHandler = DEFAULTFILEHANDLER()):
    x_scaler = ThousandScaler
    y_scaler = ThousandScaler
    scalers = dict(train_x_scaler=x_scaler, train_y_scaler=y_scaler)
    model = TrainedXASBlock.load(model_tag, file_handler=file_handler, **scalers)
    mean = MeanModel(tag=model_tag, **scalers)

    model_mse = model.metrics.median_of_mse_per_spectra
    mean_mse = mean.metrics.median_of_mse_per_spectra

    return mean_mse / model_mse


def get_universal_model_eta(data_tag: DataTag):
    universalXAS = TrainedXASBlock.load(
        tag=ModelTag(
            name="universalXAS",
            element="All",
            type="FEFF",
        ),
        train_x_scaler=ThousandScaler,
        train_y_scaler=ThousandScaler,
    )
    split = TrainedModelLoader.load_ml_splits(data_tag)
    split = ScaledMlSplit.from_splits(
        split,
        x_scaler=ThousandScaler,
        y_scaler=ThousandScaler,
    )
    universal_element_metric = universalXAS.compute_metrics(
        split
    ).median_of_mse_per_spectra
    expert_element_metric = MeanModel(
        # tag=ModelTag(name="expertXAS", **FEFFDataTags[0].dict()),
        tag=ModelTag(name="expertXAS", **data_tag.dict()),
        train_x_scaler=ThousandScaler,
        train_y_scaler=ThousandScaler,
    ).metrics.median_of_mse_per_spectra
    return expert_element_metric / universal_element_metric


def get_expert_tuned_comparision_metric(
    tag: DataTag,
    io_config: DictConfig = DEFAULTFILEHANDLER().config,
    x_scaler=ThousandScaler,
    y_scaler=ThousandScaler,
):
    file_handler = FileHandler(io_config)
    ml_splits = file_handler.deserialize_json(MLSplits, tag)
    scaled_ml_splits = ScaledMlSplit.from_splits(ml_splits, x_scaler, y_scaler)

    load_model = partial(
        TrainedXASBlock.load,
        **dict(
            train_x_scaler=x_scaler,
            train_y_scaler=y_scaler,
            file_handler=file_handler,
        ),
    )
    expert_tag = ModelTag(name="expertXAS", **tag.dict())
    tuned_tag = ModelTag(name="tunedUniversalXAS", **tag.dict())

    scaled_ml_splits = TrainedModelLoader.load_scaled_splits(
        expert_tag, x_scaler, y_scaler, file_handler
    )

    metric1 = load_model(expert_tag).compute_metrics(scaled_ml_splits)
    metric2 = load_model(tuned_tag).compute_metrics(scaled_ml_splits)
    return ComparisonMetrics(metric1=metric1, metric2=metric2)


# %%
