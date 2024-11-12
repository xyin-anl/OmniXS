# %%

from functools import partial

import yaml
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from omnixas.data import (
    DataTag,
    ElementsFEFF,
    MLSplits,
)
from omnixas.model.trained_model import (
    MeanModel,
    ModelMetrics,
    ModelTag,
    TrainedModelLoader,
    TrainedXASBlock,
)
from omnixas.scripts.plots.scripts import ExpertEtas
from omnixas.utils import DEFAULTFILEHANDLER

# %%


def ml_model_etas(feature_name, ml_model):
    out = {}
    for element in ElementsFEFF:
        split = DEFAULTFILEHANDLER().deserialize_json(
            MLSplits,
            DataTag(element=element, type="FEFF", feature=feature_name),
        )
        reg = ml_model.fit(split.train.X, split.train.y)
        predictions = reg.predict(split.test.X)
        metrics = ModelMetrics(predictions=predictions, targets=split.test.y)
        out[element] = (
            MeanModel(
                tag=ModelTag(element=element, type="FEFF", name="meanmodel"),
            ).metrics.median_of_mse_per_spectra
            / metrics.median_of_mse_per_spectra
        )
    return out


class DscribeNetEtas:

    def __new__(
        cls,
        feature_name: str,
        **kwargs,
    ):

        file_name = f"../../config/training/{feature_name.lower()}.yaml"
        dims = DictConfig(yaml.load(open(file_name), Loader=yaml.FullLoader))[
            feature_name
        ].dim

        return {
            element: MeanModel(
                tag=ModelTag(element=element, type="FEFF", name="meanmodel"),
            ).metrics.median_of_mse_per_spectra
            / TrainedXASBlock(
                tag=ModelTag(
                    element=element,
                    type="FEFF",
                    feature=feature_name,
                    name=f"{feature_name.upper()}Net",
                ),
                model=TrainedModelLoader.load_model(
                    tag=ModelTag(
                        element=element,
                        type="FEFF",
                        feature=feature_name,
                        name=f"{feature_name.upper()}Net",
                    ),
                    input_dim=dims[element],
                ),
            )
            .compute_metrics(
                TrainedModelLoader.load_scaled_splits(
                    ModelTag(
                        element=element,
                        type="FEFF",
                        feature=feature_name,
                        name=f"{feature_name.upper()}Net",
                    )
                )
            )
            .median_of_mse_per_spectra
            for element in ElementsFEFF
        }


ACSFNetEtas = partial(DscribeNetEtas, feature_name="ACSF")
SOAPNetEtas = partial(DscribeNetEtas, feature_name="SOAP")


# %%

if __name__ == "__main__":
    print(f"ACSF MLP: {ACSFNetEtas()}")
    print(f"SOAP MLP: {SOAPNetEtas()}")
    print(ExpertEtas())

    print(ml_model_etas("ACSF", LinearRegression()))

    print("SOAP (Linear Regression)")
    print(ml_model_etas("SOAP", LinearRegression()))

    print("Transfer-feature (Linear Regression)")
    print(ml_model_etas("m3gnet", LinearRegression()))

    print("ACSF XGBRegressor")
    print(ml_model_etas("ACSF", XGBRegressor()))

    print("SOAP XGBRegressor")
    print(ml_model_etas("SOAP", XGBRegressor()))

    print("Transfer-feature (XGBRegressor)")
    print(ml_model_etas("m3gnet", XGBRegressor()))
