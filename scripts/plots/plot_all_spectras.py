import os
from functools import cached_property
from typing import Literal

import numpy as np
import scienceplots
from matplotlib import pyplot as plt

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data
from utils.src.plots.heatmap_of_lines import heatmap_of_lines


class MLDATAPlotter:

    def __init__(
        self,
        compound_name,
        simulation_type,
        ax: plt.Axes = plt.gca(),
        fontsize=12,
        split: Literal["all", "train", "val", "test"] = "all",
    ):
        self.compound = compound_name
        self.simulation_type = simulation_type
        self.ax = ax
        self.fontsize = fontsize
        self.subset = split

    @staticmethod
    def stylize(fn):
        def wrapper(*args, **kwargs):
            plt.style.use(["default", "science"])
            result = fn(*args, **kwargs)
            plt.style.use("default")
            return result

        return wrapper

    def add_texts(self, title, x_label, y_label):
        title += f": {self.compound} {self.simulation_type}"
        self.ax.set_title(title, fontsize=self.fontsize)
        self.ax.set_xlabel(x_label, fontsize=self.fontsize)
        self.ax.set_ylabel(y_label, fontsize=self.fontsize)

    @stylize
    def plot_spectra_heatmap(self):
        title = f"{self.subset.capitalize()} spectras"
        self.add_texts(title, "Energy (eV)", "Spectra (Arbitrary Units)")
        heatmap_of_lines(self.spectras, ax=self.ax, x_ticks=self.energy_ticks)
        return self

    @property
    def energy_ticks(self):
        energy = self.energies
        energy_ticks = np.linspace(energy.min(), energy.max(), 5)
        energy_ticks = np.round(energy_ticks, 1)
        return energy_ticks

    @cached_property
    def spectras(self):
        if self.subset == "all":
            all_spectras = [
                self.splits.train.y,
                self.splits.val.y,
                self.splits.test.y,
            ]
            return np.concatenate(all_spectras)
        else:
            return self.splits[self.subset].y

    @stylize
    def plot_feature_heatmap(self):
        title = f"{self.subset.capitalize} features"
        self.add_texts(title, "Index", "Feature")
        heatmap_of_lines(self.features(self.subset), ax=self.ax)
        return self

    def features(self):
        if self.subset == "all":
            all_features = [
                self.splits.train.X,
                self.splits.val.X,
                self.splits.test.X,
            ]
            return np.concatenate(all_features)
        else:
            return self.splits[type].X

    @property
    def energies(self):
        path = cfg.paths.ml_data.format(
            compound=self.compound,
            simulation_type=self.simulation_type,
        )
        energies = np.load(path)["energies"]
        return energies

    @cached_property
    def splits(self):
        query = DataQuery(self.compound, self.simulation_type)
        return load_xas_ml_data(query)

    def _set_energy_ticks(self):
        loc = np.linspace(0, len(self.energies), 5)
        labels = np.linspace(self.energies.min(), self.energies.max(), 5)
        self.ax.set_xticks(loc, labels)

    def plot_average_spectra(self):
        title = "Mean Spectra"
        legend = f"Mean +- Std of {self.compound} {self.simulation_type}"
        self.add_texts(title, "Energy (eV)", "Spectra (Arbitrary Units)")
        self.ax.plot(self.spectras.mean(axis=0), label=legend)
        self.ax.fill_between(
            range(len(self.energies)),
            self.spectras.mean(axis=0) - self.spectras.std(axis=0),
            self.spectras.mean(axis=0) + self.spectras.std(axis=0),
            alpha=0.3,
        )
        self._set_energy_ticks()
        return self

    def legend(self):
        self.ax.legend(fontsize=self.fontsize, loc="lower right")
        return self

    def save(self, path):
        if path is None:
            path = f"{self.compound}_{self.simulation_type}.pdf"
        self.ax.figure.savefig(path)


if __name__ == "__main__":
    compounds = ["Cu", "Ti"]
    simulation_types = ["FEFF", "VASP"]

    # for compound in compounds:
    #     for simulation_type in simulation_types:
    #         heatmap_plotter = MLDATAPlotter(compound, simulation_type)
    #         heatmap_plotter.plot_spectra_heatmap().save(
    #             f"{compound}_{simulation_type}_heatmap.pdf"
    #         )
    #         heatmap_plotter.ax.clear()

    for compound in compounds:
        for simulation_type in simulation_types:
            feature_plotter = MLDATAPlotter(
                compound, simulation_type
            ).plot_average_spectra()
        feature_plotter.legend().save(f"{compound}_average_spectra.pdf")
        feature_plotter.ax.clear()
