from omnixas.data import DataTag, ScaledMlSplit, ThousandScaler
from omnixas.model.trained_model import (
    MeanModel,
    ModelTag,
    TrainedModelLoader,
    TrainedXASBlock,
)
from omnixas.utils.io import DEFAULTFILEHANDLER, FileHandler


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
