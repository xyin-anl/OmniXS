import os
import pickle
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Union

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.ml_models import MeanModel


class TrainedModel(ABC):  #
    def __init__(self, query: DataQuery):
        self.compound = query.element
        self.simulation_type = query.simulation
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
    def median_relative_to_mean_model(self):
        return (
            MeanModel(query=self.query).median_of_mse_per_spectra
            / self.median_of_mse_per_spectra
        )

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
        sort_array = sort_array or self.mse_per_spectra
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
        self.__dict__.pop("median_relative_to_mean_model", None)

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