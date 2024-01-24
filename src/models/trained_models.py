import os
from hydra.utils import instantiate
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import torch
import optuna
from main import FC_XAS


from functools import cached_property
from config.defaults import cfg

from src.data.ml_data import DataQuery, load_xas_ml_data

from sklearn.metrics import mean_absolute_error, mean_squared_error


class TrainedModel(ABC):
    def __init__(self, query: DataQuery):
        self.compound = query.compound
        self.simulation_type = query.simulation_type
        self.query = query

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

    @cached_property
    def mse(self):
        return mean_squared_error(self.data.test.y, self.predictions, squared=True)

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
        max_idx = np.argmax(self.data.test.y, axis=1)
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


class Trained_FCModel(TrainedModel):
    name = "FCModel"

    def __init__(self, query, date_time=None, version=None, ckpt_name="last"):
        super().__init__(query)
        self.date_time = date_time or self._latest_dir(self._hydra_dir)
        self.version = (
            version or self._latest_dir(self._lightning_log_dir).split("_")[-1]
        )  # TODO: make it try optuna study
        self.ckpt_name = ckpt_name

    @cached_property
    def model(self):
        # model = instantiate(cfg.model)
        model = FC_XAS(widths=[64, 100, 141])
        model_params = torch.load(self._ckpt_path)
        # change keys of state_dict to remove the "model." prefix
        model_params["state_dict"] = {
            k.replace("model.", ""): v for k, v in model_params["state_dict"].items()
        }
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
            key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True
        )
        return dirs[0]

    @property
    def _hydra_dir(self):
        dir = "logs/{compound}-{simulation_type}/runs/".format(**self.query.__dict__)
        assert os.path.exists(dir), f"Hydra dir {dir} not found"
        return dir

    @property
    def _lightning_log_dir(self):
        lightning_dir = self._hydra_dir + self.date_time + "/lightning_logs/"
        assert os.path.exists(lightning_dir), f"lightning_dir {lightning_dir} not found"
        return lightning_dir

    @cached_property
    def _ckpt_path(self, version=None):
        log_dir = self._lightning_log_dir
        version_dir = f"version_{self.version}"
        ckpt_path = log_dir + version_dir + f"/checkpoints/{self.ckpt_name}.ckpt"
        assert os.path.exists(ckpt_path), f"ckpt_path {ckpt_path} not found"
        return ckpt_path
