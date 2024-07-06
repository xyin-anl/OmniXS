# %%
import warnings
from config.defaults import cfg
from dataclasses import dataclass
from functools import cached_property, lru_cache

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch

from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import MeanModel, Trained_FCModel

# %%


@dataclass
class Model:
    name: str
    compound: str
    simulation_type: str

    def __post_init__(self):
        if self.name == "universal_tl" and self.compound != "ALL":
            warn = f"Model {self.name} is trained on all compounds. Setting compound to 'ALL' to avoid errors."
            warnings.warn(warn)
            self.compound = "ALL"

    def __hash__(self):
        return hash((self.name, self.compound, self.simulation_type))

    def __eq__(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        return (self.name, self.compound, self.simulation_type) == (
            other.name,
            other.compound,
            other.simulation_type,
        )

    @dataclass
    class SpectraPair:
        pred: np.ndarray
        true: np.ndarray

    @property
    def model(self):
        if self.name != "MeanModel":
            return Trained_FCModel(
                DataQuery(self.compound, self.simulation_type), name=self.name
            )
        else:
            return MeanModel(DataQuery(self.compound, self.simulation_type))

    @lru_cache
    def predict(self, compound=None, simulation_type=None):
        compound = compound or self.compound
        simulation_type = simulation_type or self.simulation_type
        data = load_xas_ml_data(DataQuery(compound, simulation_type)).test
        features = data.X
        spectras = data.y
        return self.SpectraPair(pred=self.model(features), true=spectras)


@dataclass
class PredError:
    pair: Model.SpectraPair

    @property
    def mse(self):
        return np.mean((self.pair.pred - self.pair.true) ** 2)

    @property
    def mse_per_energy(self):
        return np.mean((self.pair.pred - self.pair.true) ** 2, axis=0)

    @cached_property
    def e0(self):
        return np.array([np.gradient(spectra).argmax() for spectra in self.pair.true])

    @property
    def pre_e0_mses(self):
        return np.array(
            [
                np.mean((spectra[:e0] - prediction[:e0]) ** 2, axis=0)
                for spectra, prediction, e0 in zip(
                    self.pair.true, self.pair.pred, self.e0
                )
            ]
        )

    @property
    def pre_e0_mse(self):
        return np.mean(self.pre_e0_mses)

    @property
    def post_e0_mses(self):
        return np.array(
            [
                np.mean((spectra[e0:] - prediction[e0:]) ** 2)
                for spectra, prediction, e0 in zip(
                    self.pair.true, self.pair.pred, self.e0
                )
            ]
        )

    @property
    def post_e0_mse(self):
        return np.mean(self.post_e0_mses)

    @cached_property
    def true_derivative(self):
        return np.array([np.gradient(spectra) for spectra in self.pair.true])

    @cached_property
    def pred_derivative(self):
        return np.array([np.gradient(spectra) for spectra in self.pair.pred])

    @property
    def mse_derivative(self):
        return np.mean((self.pred_derivative - self.true_derivative) ** 2)

    @property
    def mse_derivative_per_energy(self):
        return np.mean((self.pred_derivative - self.true_derivative) ** 2, axis=0)

    @property
    def std_mse_per_energy(self):
        return np.std((self.pair.pred - self.pair.true) ** 2, axis=0)


# %%


def compare_mse_per_energy(
    model_names, compounds, simulation_type="FEFF", axs=None, fontsize=18
):
    if axs is None:
        fig, axs = plt.subplots(
            len(compounds), 1, figsize=(12, 4 * len(compounds)), sharex=True
        )
    axs = axs.flatten()
    compound_colors = {c: plt.cm.tab10.colors[i] for i, c in enumerate(compounds)}
    for idx, compound in enumerate(compounds):

        def mse_per_energy(model_name):
            return PredError(
                Model(model_name, compound, simulation_type).predict(
                    compound,
                    simulation_type,
                )
            ).mse_per_energy

        mse_of_model = {
            # model_name: mse_per_energy(model_name) for model_name in model_names
            model_name: mse_per_energy("MeanModel") / mse_per_energy(model_name)
            for model_name in model_names
        }

        energy_points = np.arange(len(mse_of_model[model_names[0]]))

        LINEWIDTH = 0.4
        axs[idx].plot(
            energy_points,
            mse_of_model["ft_tl"],
            color="green",
            linewidth=LINEWIDTH,
            # label=r"$\eta_{E}^{Tuned-Universal}$",
            label="Tuned-Universal",
        )

        axs[idx].plot(
            energy_points,
            mse_of_model["per_compound_tl"],
            color="red",
            linewidth=LINEWIDTH,
            # label=r"$\eta_{E}^{Expert}$",
            label="Expert",
        )

        axs[idx].fill_between(
            energy_points,
            mse_of_model["ft_tl"],
            mse_of_model["per_compound_tl"],
            color="green",
            where=mse_of_model["ft_tl"] >= mse_of_model["per_compound_tl"],
            alpha=0.5,
            label="Tuned-universal is better",
        )

        axs[idx].fill_between(
            energy_points,
            mse_of_model["ft_tl"],
            mse_of_model["per_compound_tl"],
            color="red",
            where=mse_of_model["ft_tl"] < mse_of_model["per_compound_tl"],
            alpha=0.5,
            label="Expert is better",
        )

        axs[idx].text(
            0.94,
            0.9,
            compound,
            fontsize=fontsize,
            transform=axs[idx].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor=compound_colors[compound], alpha=0.2, edgecolor="white"
            ),
        )
        axs[idx].set_xlim(energy_points[0], energy_points[-1])

        axs[idx].set_xticks([])

        axs[idx].yaxis.set_tick_params(which="both", right=False)

        axs[idx].yaxis.set_tick_params(which="minor", right=False)


model_names = [
    "per_compound_tl",
    "ft_tl",
    "universal_tl",
]

fig = plt.figure(figsize=(8, 10))
plt.style.use(["default", "science"])
gs = fig.add_gridspec(8, 1, hspace=0.0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
FONTSIZE = 18
compare_mse_per_energy(model_names, cfg.compounds, axs=axs, fontsize=FONTSIZE)

#  legend outide the top axes
axs[0].legend(
    fontsize=FONTSIZE * 0.8,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.7),
    ncol=2,
    frameon=False,
)
axs[-1].set_xlabel(r"$\Delta E$ (0.25 eV)", fontsize=FONTSIZE)
axs[-1].set_xticks(np.arange(0, 141, 20))

# axs[4].set_ylabel(r"$\eta_{E}$", fontsize=FONTSIZE * 1.2)

# add label on center of figure
fig.text(
    0,
    0.5,
    r"$\eta_{E}$",
    va="center",
    rotation="vertical",
    fontsize=FONTSIZE * 1.5,
    ha="center",
)
fig.tight_layout()
fig.savefig("performance_across_energy.pdf", bbox_inches="tight", dpi=300)

# %%
