import warnings
import os
import pickle
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Literal, Union

import numpy as np
import optuna
import torch
from hydra.utils import instantiate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor

# multioutput regressor
from sklearn.multioutput import MultiOutputRegressor

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.xas_fc import FC_XAS


class PreTrainedFCXASModel(FC_XAS):
    # Thin Wrapper around Trained_FCModel.model to allow for fine-tuning
    def __init__(
        self,
        query: DataQuery,
        name,
        **model_kwargs,
    ):
        trained_model = Trained_FCModel(query=query, name=name).model

        # # TODO: skipped as fc_xas attributes were later changed
        # for k, v in model_kwargs.items():
        #     if not hasattr(trained_model, k):
        #         raise ValueError(f"model does not have attribute {k}")
        #     if getattr(trained_model, k) != v:
        #         raise ValueError(
        #             f"model attribute {k} is not equal to {v} instead {getattr(trained_model, k)}"
        #         )

        super().__init__(**model_kwargs)
        self.load_state_dict(trained_model.state_dict())


class TrainedModel(ABC):  #
    def __init__(self, query: DataQuery):
        self.compound = query.compound
        self.simulation_type = query.simulation_type
        self.query = query
        self._data = load_xas_ml_data(query=query)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predictions(self):
        pass

    def __call__(self, X: Union[np.ndarray, torch.Tensor]):
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        return self.model(X).detach().numpy()

    @property
    def _cache_dir(self):
        path = cfg.paths.cache.ml_models.format(
            compound=self.compound, simulation_type=self.simulation_type
        )
        path, ext = os.path.splitext(path)
        path += f"_{self.name}{ext}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def cache_trained_model(self, path=None):
        if path is None:
            path = self._cache_dir
            warnings.warn(f"Saving model to default path {path}")

        assert not isinstance(
            self.model, torch.nn.Module
        ), "nn.Module save not supported. Use torch.save"
        self.mse  # proxy to train
        pickle.dump(self.model, open(path, "wb"))
        return self

    def load(self, path=None):
        if path is None:
            path = self._cache_dir
            warnings.warn(f"Loading model from default path {path}")
        self._clear_cached_properties()
        self.model = pickle.load(open(path, "rb"))
        return self

    @cached_property
    def median_of_mse_per_spectra(self):
        return np.median(self.mse_per_spectra)

    @cached_property
    def mae(self):
        return mean_absolute_error(self.data.test.y, self.predictions)

    @cached_property
    def mae_per_spectra(self):
        return np.mean(np.abs(self.data.test.y - self.predictions), axis=1)

    @cached_property
    def mse(self):
        return mean_squared_error(self.data.test.y, self.predictions, squared=True)

    @cached_property
    def mse_per_spectra(self):
        return np.mean((self.data.test.y - self.predictions) ** 2, axis=1)

    @cached_property
    def geometric_mean_of_mse_per_spectra(self):
        return np.exp(np.mean(np.log(self.mse_per_spectra)))

    @cached_property
    def gmean_ratio_to_mean_model(self):
        return (
            MeanModel(query=self.query).geometric_mean_of_mse_per_spectra
            / self.geometric_mean_of_mse_per_spectra
        )

    @cached_property
    def mse_relative_to_mean_model(self):
        return MeanModel(query=self.query).mse / self.mse

    def sorted_predictions(self, sort_array=None):
        sort_array = sort_array or self.mae_per_spectra  # default sort by mae
        pair = np.column_stack((self.data.test.y, self.predictions))
        pair = pair.reshape(-1, 2, self.data.test.y.shape[1])
        pair = pair[sort_array.argsort()]
        return pair, sort_array.argsort()

    def top_predictions(self, splits=10, drop_last_split=True):
        # sort by mean residue, splits and return top of each split
        pair = self.sorted_predictions()[0]
        # for even split, some pairs are chopped off
        new_len = len(pair) - divmod(len(pair), splits)[1]
        pair = pair[:new_len]

        top_spectra = [s[-1] for s in np.split(pair, splits)]  # bottom of each split
        if drop_last_split:
            top_spectra = top_spectra[:-1]

        return np.array(top_spectra)

    @cached_property
    def absolute_errors(self):
        return np.abs(self.data.test.y - self.predictions)

    @property
    def data(self):
        return self._data

    def _clear_cached_properties(self):
        self.__dict__.pop("predictions", None)
        self.__dict__.pop("mae_per_spectra", None)
        self.__dict__.pop("mse_per_spectra", None)
        self.__dict__.pop("mae", None)
        self.__dict__.pop("mse", None)
        self.__dict__.pop("absolute_errors", None)
        self.__dict__.pop("sorted_predictions", None)
        self.__dict__.pop("top_predictions", None)
        self.__dict__.pop("mse_relative_to_mean_model", None)
        self.__dict__.pop("peak_errors", None)
        self.__dict__.pop("r2", None)
        self.__dict__.pop("geometric_mean_of_mse_per_spectra", None)
        self.__dict__.pop("gmean_ratio_to_mean_model", None)
        self.__dict__.pop("median_of_mse_per_spectra", None)

    @data.setter
    def data(self, data):
        self._clear_cached_properties()
        self._data = data

    @cached_property
    def peak_errors(self):
        max_idx = np.argmax(self.data.test.y, axis=1)
        peak_errors = np.array(
            [error[idx] for error, idx in zip(self.model.absolute_errors, max_idx)]
        )
        return peak_errors

    @property
    def r2(self):
        return r2_score(self.data.test.y, self.predictions)


class MeanModel(TrainedModel):
    name = "MeanModel"

    @property
    def model(self):
        return lambda *args, **kwargs: torch.Tensor(self.predictions)

    @cached_property
    def predictions(self):
        return np.array([np.mean(self.data.train.y, axis=0)] * len(self.data.test.y))


class LinReg(TrainedModel):
    def __init__(self, query: DataQuery, name="LinReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return LinearRegression().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class RidgeReg(TrainedModel):
    def __init__(self, query: DataQuery, name="RidgeReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return Ridge().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class SVReg(TrainedModel):
    def __init__(self, query: DataQuery, name="SVReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return MultiOutputRegressor(estimator=SVR()).fit(
            self.data.train.X, self.data.train.y
        )

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class RFReg(TrainedModel):
    def __init__(self, query: DataQuery, name="RFReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return RandomForestRegressor().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class XGBReg(TrainedModel):
    def __init__(self, query: DataQuery, name="XGBReg"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return XGBRegressor().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class ElastNet(TrainedModel):

    def __init__(self, query: DataQuery, name="ElasticNet"):
        super().__init__(query)
        self._name = name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        return ElasticNet().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)


class Trained_FCModel(TrainedModel):

    def __init__(self, query, name, date_time=None, version=None, ckpt_name=None):
        super().__init__(query)
        self._name = name
        self.date_time = date_time or self._latest_dir(self._hydra_dir)
        self.version = (
            version or self._latest_dir(self._lightning_log_dir).split("_")[-1]
        )  # TODO: make it try optuna study
        self.ckpt_name = ckpt_name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        # model = instantiate(cfg.model)

        model_params = torch.load(self._ckpt_path)
        # change keys of state_dict to remove the "model." prefix
        model_params["state_dict"] = {
            k.replace("model.", ""): v for k, v in model_params["state_dict"].items()
        }

        # infer widths from state_dict
        # TODO: double check this
        state = model_params["state_dict"]
        weight_shapes = [
            state[k].shape for k in state.keys() if "layers" in k and "weight" in k
        ]

        input_sz = weight_shapes[0][1]
        hidden_sz = [w[0] for w in weight_shapes[:]]
        output_sz = weight_shapes[-1][0]
        widths = [input_sz] + hidden_sz  # output is already included

        model = FC_XAS(widths=widths, input_dim=None, output_dim=None)

        # model = FC_XAS(
        #     widths=hidden_sz,
        #     input_dim=None,
        #     output_dim=None,
        #     compound=self.compound,
        #     simulation_type=self.simulation_type,
        # )

        model.load_state_dict(model_params["state_dict"])
        model.eval()
        return model

    @cached_property
    def optuna_study(self):
        kwargs = dict(compound=self.compound, simulation_type=self.simulation_type)
        # TODO: move this config to yaml
        study_name = f"{self.compound}-{self.simulation_type}"
        storage = cfg.paths.optuna_db.format(**kwargs)
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study

    @cached_property
    def predictions(self):
        return self.model(torch.Tensor(self.data.test.X)).detach().numpy()

    def _latest_dir(self, directory):
        assert os.path.exists(directory), f"Directory {directory} does not exist"
        all_items = os.listdir(directory)
        dirs = [  # Filter out items that are not directories
            item for item in all_items if os.path.isdir(os.path.join(directory, item))
        ]
        assert len(dirs) > 0, f"Directory {directory} is empty"
        dirs.sort(  # Sort directories by creation time
            key=lambda x: os.path.getctime(os.path.join(directory, x)),
            reverse=True,
        )
        return dirs[0]

    @property
    def _hydra_dir(self):
        hydra_dir = "logs/{compound}_{simulation_type}/runs/".format(
            **self.query.__dict__
        )
        hydra_dir = hydra_dir.replace("runs", self.name)  # adhoc TODO:
        assert os.path.exists(hydra_dir), f"Hydra dir {hydra_dir} not found"
        return hydra_dir

    @property
    def _lightning_log_dir(self):
        lightning_dir = self._hydra_dir + self.date_time + "/lightning_logs/"
        assert os.path.exists(lightning_dir), f"lightning_dir {lightning_dir} not found"
        return lightning_dir

    @cached_property
    def _ckpt_path(self, version=None):
        log_dir = self._lightning_log_dir
        version_dir = f"version_{self.version}"
        # if ckpt path is none select one that starts with "epoch*.ckpt"

        # ckpt_path = log_dir + version_dir + f"/checkpoints/{self.ckpt_name}.ckpt"
        ckpt_path = log_dir + version_dir + "/checkpoints/"
        if self.ckpt_name is None:
            ckpt_path += [f for f in os.listdir(ckpt_path) if f.startswith("epoch")][0]
        else:
            ckpt_path += self.ckpt_name + ".ckpt"
        assert os.path.exists(ckpt_path), f"ckpt_path {ckpt_path} not found"
        return ckpt_path


if __name__ == "__main__":
    # model = Trained_FCModel(DataQuery("ALL", "FEFF")).mse
    # model = PreTrainedFCXASModel(DataQuery("Cu", "FEFF"))

    # TESTING SAVE AND LOAD
    model = LinReg(DataQuery("Cu", "FEFF"))
    print(model.mse)
    model.cache_trained_model()
    print("saved")
    model = model.load()
    print("loaded")
    print(model.mse)
    model2 = LinReg(DataQuery("Cu", "FEFF"))
    model2.load()
