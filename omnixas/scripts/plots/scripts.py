# %%
from functools import partial

import numpy as np
from typing import Dict, Tuple
from omnixas.data import AllDataTags, DataTag, ScaledMlSplit, ThousandScaler
from omnixas.model.trained_model import (
    ComparisonMetrics,
    MeanModel,
    ModelTag,
    TrainedModelLoader,
    TrainedXASBlock,
)


class XASModelsOfCategory:
    def __new__(cls, model_name: str, **kwargs):
        return {
            ModelTag(name=model_name, **tag.dict()): TrainedXASBlock.load(
                ModelTag(name=model_name, **tag.dict()),
                **kwargs,
            )
            for tag in AllDataTags()
        }


class AllMlSplits:
    def __new__(cls, x_scaler=ThousandScaler, y_scaler=ThousandScaler, **kwargs):
        return {
            tag: ScaledMlSplit.from_splits(
                TrainedModelLoader.load_ml_splits(tag),
                x_scaler=x_scaler,
                y_scaler=y_scaler,
            )
            for tag in AllDataTags()
        }


class MetricsForModelCategory:
    def __new__(cls, model_name: str, **kwargs):
        return {
            model.tag: model.compute_metrics(splits)
            for splits, model in zip(
                AllMlSplits(**kwargs).values(),
                XASModelsOfCategory(model_name, **kwargs).values(),
            )
        }


AllExpertModels = partial(XASModelsOfCategory, model_name="expertXAS")
AllTunedModels = partial(XASModelsOfCategory, model_name="tunedUniversalXAS")


class AllXASModels:
    def __new__(cls, **kwargs):
        expert_models = AllExpertModels(**kwargs)
        tuned_models = AllTunedModels(**kwargs)
        return {**expert_models, **tuned_models}


AllExpertMetrics = partial(MetricsForModelCategory, model_name="expertXAS")
AllTunedMetrics = partial(MetricsForModelCategory, model_name="tunedUniversalXAS")


class CompareAllExpertAndTuned:
    def __new__(cls, **kwargs):  # kwargs can incldue file_handler, x_scaler, y_scaler
        out = {}
        for m1, m2 in zip(
            AllExpertMetrics(**kwargs).items(),
            AllTunedMetrics(**kwargs).items(),
        ):
            out[m1[0]] = ComparisonMetrics(metric1=m1[1], metric2=m2[1])
        return out


class ExpertAndTunedEtas:
    def __new__(cls, **kwargs) -> dict:
        all_models = AllXASModels(**kwargs)
        return {
            model_tag: (
                MeanModel(tag=model_tag, **kwargs).metrics.median_of_mse_per_spectra
                / model.metrics.median_of_mse_per_spectra
            )
            for model_tag, model in all_models.items()
        }


class UniversalModelEtas:
    def __new__(cls, **kwargs):
        universal_model = TrainedXASBlock.load(
            tag=ModelTag(
                name="universalXAS",
                element="All",
                type="FEFF",
            ),
            **kwargs,
        )
        return {
            ModelTag(name="universalXAS", **tag.dict()): (
                MeanModel(
                    tag=ModelTag(name="meanmodel", **tag.dict()), **kwargs
                ).metrics.median_of_mse_per_spectra
                / universal_model.compute_metrics(
                    ScaledMlSplit.from_splits(
                        TrainedModelLoader.load_ml_splits(tag),
                        x_scaler=kwargs["x_scaler"],
                        y_scaler=kwargs["y_scaler"],
                    )
                ).median_of_mse_per_spectra
            )
            for tag in AllDataTags()
        }


class AllEtas:
    def __new__(cls, **kwargs):
        return {**ExpertAndTunedEtas(**kwargs), **UniversalModelEtas(**kwargs)}


class ExpertTunedWinRates:
    def __new__(cls, **kwargs) -> Dict[ModelTag, Tuple[float, float]]:
        comparison_metrics = CompareAllExpertAndTuned(**kwargs)
        win_rates = {}
        for tag, metrics in comparison_metrics.items():
            differences = metrics.residual_diff
            energy_rate = float(np.mean(np.mean(differences, axis=0) > 0) * 100)
            spectra_rate = float(np.mean(np.mean(differences, axis=1) > 0) * 100)
            win_rates[tag] = dict(energy=energy_rate, spectra=spectra_rate)
        return win_rates


# %%

if __name__ == "__main__":
    from omnixas.utils import DEFAULTFILEHANDLER

    etas = AllEtas(
        x_scaler=ThousandScaler,
        y_scaler=ThousandScaler,
        file_handler=DEFAULTFILEHANDLER(),
    )
    print(etas)

    win_rates = ExpertTunedWinRates(
        x_scaler=ThousandScaler,
        y_scaler=ThousandScaler,
        file_handler=DEFAULTFILEHANDLER(),
    )
    print(win_rates)

# %%
