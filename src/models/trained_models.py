import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


from functools import cached_property

from src.data.ml_data import DataQuery, load_xas_ml_data

from sklearn.metrics import mean_absolute_error


class TrainedModel(ABC):
    def __init__(self, compound, simulation_type="FEFF"):
        self.compound = compound
        self.simulation_type = simulation_type

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def predictions(self):
        pass

    @cached_property
    def mae_per_spectra(self):
        return np.mean(np.abs(self.data.test.y - self.predictions), axis=1)

    @cached_property
    def mae(self):
        return mean_absolute_error(self.data.test.y, self.predictions)

    def sorted_predictions(self, sort_array=None):
        sort_array = sort_array or self.mae_per_spectra  # default sort by mae
        pair = np.column_stack((self.data.test.y, self.predictions))
        pair = pair.reshape(-1, 2, self.data.test.y.shape[1])
        pair = pair[sort_array.argsort()]
        return pair

    def top_predictions(self, splits=10):
        # sort by mean residue, splits and return top of each split
        pair = self.sorted_predictions()
        # for even split, some pairs are chopped off
        new_len = len(pair) - divmod(len(pair), splits)[1]
        pair = pair[:new_len]
        top_spectra = [s[0] for s in np.split(pair, splits)]
        return np.array(top_spectra)

    @cached_property
    def absolute_errors(self):
        return np.abs(self.data.test.y - self.predictions)

    @cached_property
    def data(self):
        return load_xas_ml_data(
            query=DataQuery(
                compound=self.compound,
                simulation_type=self.simulation_type,
            )
        )

    @cached_property
    def peak_errors(self):
        max_idx = np.argmax(self.model.data.test.y, axis=1)
        peak_errors = np.array(
            [error[idx] for error, idx in zip(self.model.absolute_errors, max_idx)]
        )
        return peak_errors


class LinReg(TrainedModel):
    name = "LinReg"

    @cached_property
    def model(self):
        return LinearRegression().fit(self.data.train.X, self.data.train.y)

    @cached_property
    def predictions(self):
        return self.model.predict(self.data.test.X)
