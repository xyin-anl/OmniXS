# %%
from config.defaults import cfg
from matplotlib import pyplot as plt
from scripts.model_scripts.plot_universal_tl_model import universal_TL_mses
from src.data.ml_data import DataQuery
from src.models.trained_models import Trained_FCModel
from optuna.distributions import CategoricalDistribution, IntDistribution
import numpy as np
import warnings
from copy import deepcopy
from functools import cached_property
from typing import Container, Literal, Union

import optuna
from optuna import Trial, load_study
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import TrialState
from tqdm import tqdm

# %%


class OptunaExperiment:
    """abs class for optimization results with utils functions to access
    properties specific to design optmization for ai and asic"""

    def __init__(
        self,
        storage: optuna.storages.BaseStorage = JournalStorage(
            log_storage=JournalFileStorage("journal.log")
        ),
        study_id: Union[int, Literal["LATEST"]] = "LATEST",
        trial_states: Union[Container[TrialState], None] = None,
    ):
        self.storage = storage
        self._study_id = study_id
        self.trial_states = trial_states
        self._study = self._load_frozen_study()

    def _get_objectives(self, name: str):
        out = [t.user_attrs["objectives"][name] for t in self.trials]
        return np.array(out)

    @property
    def losses(self):
        return self._get_objectives("val_loss")

    @property
    def areas(self):
        return self._get_objectives("area")

    @property
    def powers(self):
        return self._get_objectives("power")

    @property
    def delays(self):
        return self._get_objectives("delay")

    def update_study_with_q_energies(self):
        self.study = load_study(
            study_name=self.name,
            storage=self.storage,
        )  # to make things mutable temporarily
        trials = tqdm(self.trials, desc="Calculating q_energies")
        q_energies = [self._trial_q_energy(t) for t in trials]
        self.study.set_system_attr("q_energies", q_energies)
        self._study = self._load_frozen_study()  # keep things frozen

    @property
    def q_energies(self):
        if self.study.system_attrs.get("q_energies") is None:
            msg = f"q_energies not found in study {self.name}.\n"
            msg += "Running update_study_with_q_energies"
            warnings.warn(msg)
            self.update_study_with_q_energies()
            return self.q_energies
        else:
            msg = f"q_energies found in study {self.name}"
            msg += "Returning cached values"
            msg += "To update run update_study_with_q_energies"
            warnings.warn(msg)
            return np.array(self.study.system_attrs["q_energies"])

    @property
    def study_id(self):
        return self.study._study_id

    def _load_frozen_study(self):
        studies = self.storage.get_all_studies()
        if self._study_id == "LATEST":
            study = studies[-1]
            self._study_id = study._study_id
            return study
        else:
            return studies[self._study_id]

    @property
    def study(self):
        return self._study

    @study.setter
    def study(self, study):
        self._study = study
        # remove cached properties
        self.__dict__.pop("trials", None)
        self.__dict__.pop("values", None)

    @cached_property
    def trials(self):
        return self.storage.get_all_trials(
            study_id=self.study_id,
            deepcopy=True,
            # deepcopy=False,
            states=self.trial_states,
        )

    @property
    def name(self):
        return self.study.study_name

    def __len__(self):
        return len(self.trials)

    @property
    def trial_durations(self):
        return [
            round(t.duration.total_seconds(), 2)
            # t.duration.total_seconds()
            for t in self.trials
            if t.state == TrialState.COMPLETE
        ]

    @property
    def duration(self):
        return round(sum(self.trial_durations), 2)

    @cached_property
    def values(self):
        return np.array([t.values for t in self.trials])

    @property
    def first_params(self):
        return {
            k: self._clean_distribution_names(v)
            for k, v in self.trials[0].distributions.items()
        }

    @property
    def distributions(self):
        unique_distributions = list(
            set([tuple(t.distributions.keys()) for t in self.trials])
        )
        if len(unique_distributions) > 1:
            warnings.warn("Trials have different distributions")
        return unique_distributions

    def _clean_distribution_names(self, dist):

        if type(dist) == CategoricalDistribution:
            return {"type": "category", "values": dist.choices}
        elif type(dist) == IntDistribution:
            return {
                "type": "log" if dist.log else "int",
                "values": [dist.low, dist.high, dist.step],
            }
        else:
            raise NotImplementedError

    def __repr__(self):

        out = "=" * 50 + "\n"
        out += f"Optimization Result for {self.name} study\n"

        out += "-" * 50 + "\n"
        out += f"Number of trials: {len(self)}\n"
        out += f"Trial States: {self.trial_states}\n"

        out += "-" * 50 + "\n"
        out += f"Duration: {self.duration} seconds\n"
        out += "=" * 50 + "\n"
        return out

    @property
    def params_batches(self):
        trial_batches = {}
        trials_list = []
        last_distribution = tuple(self.trials[0].distributions.keys())
        for t in self.trials:
            if list(t.distributions.keys()) != last_distribution:
                trial_batches[last_distribution] = trials_list
                trials_list = []
            trials_list.append(t)
            last_distribution = tuple(t.distributions.keys())
        return trial_batches


# %%


# ==============================================================================
#               COUNTS
# ==============================================================================
from config.defaults import cfg

simulation_type = "FEFF"
model_type = "per_compound_tl"
journal_dir = "hparams_journal"
counts = {}

pairs = (
    [(c, "FEFF") for c in cfg.compounds]
    + [("ALL", "FEFF")]
    + [("Ti", "VASP")]  # something wrong with this
    + [("Cu", "VASP")]
)

mses = {}
widths = {}
batch_sizes = {}
total_trials = {}
completed_trials = {}


# for c in cfg.compounds + (["ALL"] if simulation_type == "FEFF" else []):
for c, simulation_type in pairs:
    print("=" * 50)
    print(f"Loading {c}")
    print("=" * 50)
    journal_name = f"{journal_dir}/{c}_{simulation_type}_{model_type}.log"
    # print("=" * 50)
    # print(f"Loading {journal_name}")
    # print("=" * 50)
    journal = JournalStorage(log_storage=JournalFileStorage(journal_name))
    results = OptunaExperiment(storage=journal)
    mse = results.storage.get_best_trial(0).values[0]

    c = f"{c}_{simulation_type}"
    total_trials[c] = len(results.trials)

    completed_trials[c] = len(
        [t for t in results.trials if t.state == TrialState.COMPLETE]
    )

    widths[c] = [
        w for k, w in results.storage.get_best_trial(0).params.items() if "width" in k
    ]
    batch_sizes[c] = [
        b for k, b in results.storage.get_best_trial(0).params.items() if "batch" in k
    ]

    mses[f"{c}_{simulation_type[0]}"] = mse

import pandas as pd


# widths to number of parametrs in neural network
def widths_to_parm_count(widhts, input_size=64, output_size=141):
    widths = [input_size] + widhts + [output_size]
    return sum(
        [widths[i] * widths[i + 1] + widths[i + 1] for i in range(len(widths) - 1)]
    )


# table with all
df = pd.DataFrame(
    {
        "Compound": list(total_trials.keys()),
        "Total Trials": list(total_trials.values()),
        "Completed Trials": list(completed_trials.values()),
        "Batch Sizes": [x[0] for x in list(batch_sizes.values())],
        "Widths": list(widths.values()),
        "Parameters": [
            f"{widths_to_parm_count(w, input_size=64, output_size=141)/1000:.1f}k"
            # widths_to_parm_count(w, input_size=64, output_size=141)
            for w in list(widths.values())
        ],
        "MSE": list(mses.values()),
    }
)


# print with histgram background in cells along columns only
df.style.background_gradient(cmap="jet", axis=0)

print(df)

# %%

from optuna.visualization.matplotlib import plot_param_importances

plot_param_importances(
    results.study,
    params=["batch_size", "depth"],
    target_name="val_loss",
)


# %%
