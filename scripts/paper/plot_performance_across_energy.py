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
        # return np.mean((self.pair.pred - self.pair.true) ** 2, axis=0)
        return np.median((self.pair.pred - self.pair.true) ** 2, axis=0)

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


def compare_mse_per_energy(
    model_names, compounds, simulation_type="FEFF", axs=None, fontsize=18
):
    if axs is None:
        fig, axs = plt.subplots(
            len(compounds), 1, figsize=(12, 4 * len(compounds)), sharex=True
        )
    axs = axs.flatten()

    LINEWIDTH = 0.4
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
            # model_name: mse_per_energy("MeanModel") / mse_per_energy(model_name)
            model_name: mse_per_energy(model_name)
            for model_name in model_names
        }

        # energy_points = np.arange(len(mse_of_model[model_names[0]]))
        energy_points = 0.25 * np.arange(len(mse_of_model[model_names[0]]))

        # green bar for positive and red for negative in opposite direction
        differences = mse_of_model["per_compound_tl"] - mse_of_model["ft_tl"]
        improvements = np.clip(differences, a_min=0, a_max=None)
        regressions = np.clip(differences, a_min=None, a_max=0)

        baseline = MeanModel(DataQuery(compound, simulation_type)).mse

        improvements = improvements / baseline * 100
        regressions = regressions / baseline * 100

        # vlines from improvements to 0
        axs[idx].vlines(
            energy_points,
            0,
            improvements,
            color="green",
            # alpha=0.5,
        )
        axs[idx].vlines(
            energy_points,
            0,
            regressions,
            color="red",
            # alpha=0.5,
        )

        axs[idx].text(
            1 - 0.96,
            0.85,
            compound,
            fontsize=fontsize * 0.8,
            transform=axs[idx].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor=compound_colors[compound], alpha=0.2, edgecolor="white"
            ),
        )

        if simulation_type == "VASP":
            # add text bleow the compound name saying VASP with size 12
            axs[idx].text(
                1 - 0.97,
                0.45,
                "VASP",
                fontsize=10,
                color="#101010",
                transform=axs[idx].transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(facecolor="white", alpha=0.2, edgecolor="white"),
            )

        axs_mean_spectra = axs[idx].twinx()
        mean_spectra = np.mean(
            load_xas_ml_data(DataQuery(compound, simulation_type)).test.y, axis=0
        )
        mean_spectra = mean_spectra / 1000  # TODO: remove hardcoding
        axs_mean_spectra.plot(
            energy_points,
            mean_spectra,
            color="#101010",
            linewidth=LINEWIDTH * 2,
            linestyle=":",
            label="Mean spectra",
            zorder=0,
        )
        axs_mean_spectra.spines["right"].set_color("gray")
        axs_mean_spectra.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs_mean_spectra.tick_params(axis="y", which="minor", length=0)

        axs[idx].axhline(0, color="black", linestyle="-", linewidth=0.5)
        axs[idx].tick_params(axis="y", which="both", left=True, right=False)
        axs[idx].tick_params(axis="y", which="minor", length=0)
        axs[idx].set_xlim(0, max(energy_points))

        # if compound == "Ti" and simulation_type == "FEFF":
        #     axs_mean_spectra.legend(
        #         fontsize=fontsize * 0.7,
        #         loc="upper center",
        #         frameon=False,
        #         # outside of ax on top
        #         bbox_to_anchor=(0.5, 1.5),
        #     )

        def align_yaxis_zero(ax1, ax2):
            """Align y=0 of two axes."""
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()

            prop1 = abs(y1_min) / (abs(y1_min) + abs(y1_max))
            prop2 = abs(y2_min) / (abs(y2_min) + abs(y2_max))

            ax2.set_ylim([-(y2_max - y2_min) * prop1 / (1 - prop1), y2_max])

            prop1 = abs(y1_min) / (abs(y1_min) + abs(y1_max))
            prop2 = abs(y2_min) / (abs(y2_min) + abs(y2_max))

            ax2.set_ylim([-(y2_max - y2_min) * prop1 / (1 - prop1), y2_max])

        align_yaxis_zero(axs[idx], axs_mean_spectra)


model_names = [
    "per_compound_tl",
    "ft_tl",
    "universal_tl",
]

fig = plt.figure(figsize=(5, 15))
plt.style.use(["default", "science"])
gs = fig.add_gridspec(10, 1)  # , hspace=0.2, wspace=0)
axs = gs.subplots(sharex=True, sharey=False)
FONTSIZE = 18

compare_mse_per_energy(model_names, cfg.compounds, axs=axs, fontsize=FONTSIZE)


axs[0].legend(
    fontsize=FONTSIZE * 0.7,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.8),
    ncol=2,
    frameon=False,
)
axs[-1].set_xlabel(r"$\Delta E$ (eV)", fontsize=FONTSIZE * 0.9)

# axs[-1].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

# axs[-1].tick_params(axis="x", which="major", length=5)

# axs[4].set_ylabel(r"$\eta_{E}$", fontsize=FONTSIZE * 1.2)

# add label on center of figure
fig.text(
    0,
    0.5,
    r"$\Delta \eta_{E}" + r" \left(\%\right)$",
    va="center",
    rotation="vertical",
    fontsize=FONTSIZE * 1.1,
    ha="center",
)

# commont y label at right side of figure for all twin axses that says r"<\mu>"
fig.text(
    1,
    0.5,
    r"$\left<\mu\right>_E$",
    va="center",
    rotation=-90,
    fontsize=FONTSIZE * 1.1,
    ha="center",
)

# ADD VASP AS ELL
compare_mse_per_energy(
    ["per_compound_tl", "ft_tl"],
    ["Ti", "Cu"],
    axs=axs[-2:],
    simulation_type="VASP",
)

fig.tight_layout()
fig.savefig("performance_across_energy.pdf", bbox_inches="tight", dpi=300)

# %%

# fig = plt.figure(figsize=(8, 5))
# plt.style.use(["default", "science"])
# gs = fig.add_gridspec(2, 1, hspace=0.0, wspace=0)
# axs = gs.subplots(sharex=True, sharey=False)
# FONTSIZE = 18

# compare_mse_per_energy(
#     ["per_compound_tl", "ft_tl"],
#     ["Cu", "Ti"],
#     axs=axs,
#     fontsize=FONTSIZE,
#     simulation_type="VASP",
# )

# axs[0].legend(
#     fontsize=FONTSIZE * 0.8,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.7),
#     ncol=2,
#     frameon=False,
# )
# axs[-1].set_xlabel(r"$\Delta E$ (0.25 eV)", fontsize=FONTSIZE)
# axs[-1].set_xticks(np.arange(0, 141, 20))

# # axs[4].set_ylabel(r"$\eta_{E}$", fontsize=FONTSIZE * 1.2)

# # add label on center of figure
# fig.text(
#     0,
#     0.5,
#     r"$\eta_{E}$",
#     va="center",
#     rotation="vertical",
#     fontsize=FONTSIZE * 1.5,
#     ha="center",
# )
# fig.tight_layout()
# fig.savefig("performance_across_energy_vasp.pdf", bbox_inches="tight", dpi=300)

# # %%
