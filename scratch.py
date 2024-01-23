# %%

%load_ext autoreload
%autoreload 2

from functools import cached_property

import optuna

from config.defaults import cfg
from src.models.trained_models import TrainedModel
from importlib import reload
import sys
from utils.src.misc.icecream import ic

# %%


class Trained_FCModel(TrainedModel):
    name = "FCModel"

    @cached_property
    def model(self):
        # load from ckpt or build from params
        raise NotImplementedError

    @cached_property
    def optuna_study(self):
        kwargs = dict(
            compound=self.compound,
            simulation_type=self.simulation_type,
        )
        study = optuna.load_study(
            study_name=cfg.optuna.study_name.format(**kwargs),
            storage=cfg.paths.optuna_db.format(**kwargs),
        )
        return study

    @cached_property
    def losses(self):
        return self.optuna_study.best_value

    @cached_property
    def residues(self):
        raise NotImplementedError
    
    @cached_property
    def predictions(self):
        pass


# %%



# %%
