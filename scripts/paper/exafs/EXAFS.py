# %%

import matplotlib.pyplot as plt
import os
import pickle
import warnings
from typing import Literal

import numpy as np
import scienceplots
import torch
from scipy.fft import fft
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.preprocessing import MinMaxScaler

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import Trained_FCModel


class EXAFSSpectrum:
    def __init__(
        self,
        mu_E,
        E=None,
        compound=None,
        simulation_type=None,
    ):
        self.mu_E = mu_E
        self.compound = compound
        self.simulation_type = simulation_type

        if (E is None) and ((compound is None) or (simulation_type is None)):
            msg = "Either provide the  E-grid"
            msg += " or the compound and simulation_type to load the E-grid"
            raise ValueError(msg)

        self.E = E
        self.compound = compound
        self.simulation_type = simulation_type

        if E is None:
            self.E = self._load_E()

    @property
    def E0(self):
        """Determine the absorption edge energy E0 using first and second derivatives."""
        d_mu_dE = np.gradient(self.mu_E, self.E)
        E0_index = np.argmax(d_mu_dE)

        E0_initial = self.E[E0_index]

        # Refine E0 using the second derivative
        d2_mu_dE2 = np.gradient(d_mu_dE, self.E)
        zero_crossings = np.where(np.diff(np.sign(d2_mu_dE2)))[0]

        # Find the zero-crossing closest to the initial E0 estimate
        E0_refined = self.E[
            zero_crossings[np.abs(self.E[zero_crossings] - E0_initial).argmin()]
        ]

        return E0_refined

    @property
    def mu0_E(self):
        """Fit and return the smooth background absorption mu0(E)."""
        post_edge_mask = self.E > self.E0
        spline = UnivariateSpline(
            self.E[post_edge_mask], self.mu_E[post_edge_mask], s=1
        )
        return spline(self.E)

    @property
    def delta_mu0_E(self):
        """Return the value of mu0(E) at E0."""
        return self.mu0_E[self.E == self.E0][0]

    @property
    def X_E(self):
        """Return the normalized EXAFS signal chi(E)."""
        X_E = (self.mu_E - self.mu0_E) / self.delta_mu0_E
        X_E[self.E < self.E0] = 0
        return X_E

    @property
    def dE(self):
        diff = np.diff(self.E)
        assert np.allclose(diff, diff[0]), "E grid should be uniform"
        return diff[0]

    @property
    def dk(self):
        return np.sqrt(self.dE) / 2

    @property
    def k(self):
        return np.sqrt(self.E)

    @property
    def chi_k(self):
        E_interp = np.arange(self.E[0], self.E[-1], self.dE)
        X_E_interp = np.interp(E_interp, self.E, self.X_E)
        k_interp = np.sqrt(np.abs(E_interp - self.E0)) * np.sign(E_interp - self.E0)
        return k_interp, X_E_interp

    @property
    def chi_k2_normalized(self):

        k, chi_k = self.chi_k
        chi_k2 = chi_k * k**2
        scaler = MinMaxScaler(feature_range=(-1, 1))
        chi_k2_normalized = scaler.fit_transform(chi_k2.reshape(-1, 1)).reshape(-1)
        return k, chi_k2_normalized

    @property
    def chi_k2_fft(self):
        _, chi_k2_normalized = self.chi_k2_normalized
        window = np.hanning(len(chi_k2_normalized))  # Apply a Hanning window
        chi_k2_windowed = chi_k2_normalized * window
        chi_k2_fft = fft(chi_k2_windowed)
        R = np.fft.fftfreq(len(chi_k2_windowed), d=self.dk)
        R = np.fft.fftshift(R)
        chi_k2_fft = np.fft.fftshift(chi_k2_fft)

        chi_k2_fft = chi_k2_fft[R >= 0]
        R = R[R >= 0]

        return R, chi_k2_fft

    def _load_E(self):
        # data = load_xas_ml_data(DataQuery(self.compound, self.simulation_type))
        data = np.load(
            cfg.paths.ml_data.format(
                compound=self.compound, simulation_type=self.simulation_type
            )
        )
        return data["energies"]

    def chi_r_peak(self):
        R, chi_R = self.chi_k2_fft
        idx = np.argmax(np.abs(chi_R))
        return R[idx], chi_R[idx]

    def plot(self):
        import matplotlib.pyplot as plt

        plt.style.use(["default", "science"])

        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        ax[0, 0].plot(self.E, self.mu_E, label=r"$\mu(E)$")
        ax[0, 0].plot(
            self.E[self.E > self.E0], self.mu0_E[self.E > self.E0], label=r"$\mu_0(E)$"
        )
        ax[0, 0].set_xlabel(r"$E$ (eV)")
        ax[0, 0].set_ylabel(r"$\mu(E)$")
        ax[0, 0].axvline(
            self.E0, color="r", linestyle="--", label=r"$" + f"{self.E0:.2f}" + r"$ eV"
        )
        # twin plot of derivative
        ax0t = ax[0, 0].twinx()
        ax0t.plot(
            self.E,
            np.gradient(self.mu_E, self.E),
            color="gray",
            label=r"$d\mu/dE$",
            alpha=0.2,
        )
        ax0t.set_ylabel(r"$d\mu/dE$")
        ax0t.legend()
        ax[0, 0].legend()

        ax[0, 1].plot(self.E, self.X_E)
        ax[0, 1].set_xlabel(r"$E$ (eV)")
        ax[0, 1].set_ylabel(r"$\chi(E)$")

        # plot chi_k * k^2
        k, chi_k2 = self.chi_k2_normalized
        window = np.hanning(len(chi_k2))  # Apply a Hanning window
        chi_k2_windowed = chi_k2 * window
        ax[1, 0].plot(k, chi_k2_windowed, label="Windowed")
        ax[1, 0].set_xlabel(r"$k$ ($\AA^{-1}$)")
        ax[1, 0].set_ylabel(r"$\chi(k) \cdot k^2$")
        ax[1, 0].set_title("EXAFS signal")

        # plot after windowing
        # k, chi_k_windowed = self.chi_k2_windowed
        # ax[1, 0].plot(k, chi_k_windowed * k**2)
        k, chi_k2_normalized = self.chi_k2_normalized
        ax[1, 0].plot(k, chi_k2_normalized, label="Normalized")
        ax[1, 0].set_xlabel(r"$k$ ($\AA^{-1}$)")
        ax[1, 0].set_ylabel(r"$\chi(k) \cdot k^2$")
        ax[1, 0].set_title("EXAFS signal after windowing")
        ax[1, 0].legend()

        R, chi_R = self.chi_k2_fft

        # ax[1, 1].plot(R, np.abs(chi_R), label="Magnitude", marker="o")
        # ax[1, 1].set_xlabel(r"$R$ ($\AA$)")
        # ax[1, 1].set_ylabel(r"$|\chi(R)|$")
        # ax[1, 1].set_title("Fourier Transform of EXAFS signal")
        # print(f"R_peak: {self.chi_r_peak()}")
        # ax[1, 1].vlines(
        #     self.chi_r_peak()[0],
        #     0,
        #     np.abs(self.chi_r_peak()[1]),
        #     color="r",
        #     linestyle="--",
        #     label=r"$" + f"{self.chi_r_peak()[0]:.3f}" + r"$ $\AA$",
        #     alpha=0.5,
        # )
        # ax[1, 1].legend()

        ax1t = ax[1, 1].twinx()
        ax1t.plot(R, np.angle(chi_R), color="gray", label="Phase")
        ax1t.set_ylabel(r"$\angle \chi(R)$")
        ax1t.legend()

        plt.tight_layout()
        plt.show()


def EXAFS_compound(
    compound,
    simulation_type: Literal["FEFF", "VASP"],
    model_name: Literal["simulation", "universal", "expert", "tuned_universal"],
    # use_cache=False,
):

    # # cache file
    # filename = cfg.paths.cache.exafs.format(
    #     compound=compound,
    #     simulation_type=simulation_type,
    #     model_name=model_name,
    # )
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # if use_cache and os.path.exists(filename):
    #     warnings.warn(f"Loading existing EXAFS data from {filename}")
    #     with open(filename, "rb") as f:
    #         return np.load(f, allow_pickle=True)

    data = load_xas_ml_data(DataQuery(compound, simulation_type)).test
    spectras = data.y
    features = data.X

    if model_name == "simulation":
        preds = spectras
    elif model_name == "universal":
        preds = (
            Trained_FCModel(DataQuery("ALL", simulation_type), name="universal_tl")
            .model(torch.tensor(features))
            .detach()
            .numpy()
        )
    elif model_name == "expert":
        preds = Trained_FCModel(
            DataQuery(compound, simulation_type), name="per_compound_tl"
        ).predictions
    elif model_name == "tuned_universal":
        preds = Trained_FCModel(
            DataQuery(compound, simulation_type), name="ft_tl"
        ).predictions

    R_list = []
    exafs_list = []
    for spectra in preds:
        exafs = EXAFSSpectrum(
            spectra, compound=compound, simulation_type=simulation_type
        )
        R, chi_R = exafs.chi_k2_fft
        R_list.append(R)
        exafs_list.append(chi_R)
    # make sure all R values are same
    assert all(
        np.all(R_list[0] == R) for R in R_list
    ), "R values are not same for all EXAFS signals"

    R = R_list[0]
    chi_R = np.array(exafs_list)

    # # check if all values are same
    # exafs_list = np.array(exafs_list)  # assumed to be of same shapes
    # assert np.all(
    #     np.abs(exafs_list[:][:, 0]) == np.abs(exafs_list[::-1][:, 0])
    # ), "R values are not same"
    # R = exafs_list[0][0]
    # chi_R = np.mean(exafs_list[:][:, 1], axis=0)

    # with open(filename, "wb") as f:
    #     warnings.warn(f"{filename} will be loaded next time when use_cache=True")
    #     np.save(f, exafs_list)

    return R, chi_R


if __name__ == "__main__":
    # PLOT PROCESSING STEPS
    compound = "Cu"
    simulation_type = "FEFF"
    data = load_xas_ml_data(DataQuery(compound, simulation_type))
    spectras = data.test.y
    # np.random.seed(0)
    idx = np.random.randint(0, len(spectras))
    spectra = spectras[idx]
    exafs = EXAFSSpectrum(spectra, compound=compound, simulation_type=simulation_type)
    exafs.plot()

# %%

if __name__ == "__main__":
    # COMPUTE ALL EXAFS for element and cache

    MODEL_NAMES = [
        "simulation",
        "universal",
        "expert",
        "tuned_universal",
    ]
    exaf_data = EXAFS_compound(
        compound="Cu",
        simulation_type="FEFF",
        model_name="simulation",
    )
    R, chi_R = exaf_data
    print(f"EXAFS data shape: {R.shape}, {chi_R.shape}")
    # # CACHING
    # for compound in cfg.compounds:
    #     for simulation_type in ["FEFF"]:
    #         for model_name in MODEL_NAMES:
    #             EXAFS_compound(
    #                 compound=compound,
    #                 simulation_type=simulation_type,
    #                 model_name=model_name,
    #                 use_cache=True,  # set to false to reset
    #             )

# %%

if __name__ == "__main__":
    ## RADIAL EXAFS
    compound = "Ni"
    simulation_type = "FEFF"

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={"projection": "polar"})
    # plt.style.use(["default", "science"])

    FigPolar = plt.figure(figsize=(30, 20))
    compounds = cfg.compounds
    models = ["simulation", "expert", "universal", "tuned_universal"]
    # models = ["universal", "tuned_universal"]
    # models = ["tuned_universal"]
    base_line_model = "expert"

    # gs = FigPolar.add_gridspec(len(models), len(compounds), hspace=0, wspace=0.0)

    gs = FigPolar.add_gridspec(len(models) + 1, len(compounds), hspace=0, wspace=0.0)

    ax_all = gs.subplots(
        sharex=True,
        sharey=True,
        subplot_kw={"projection": "polar"},
    )

    model_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models)))
    model_colors = {m: c for m, c in zip(models, model_colors)}

    for axs, compound in zip(ax_all.T, compounds):
        axs[0].set_title(compound)
        for ax, model in zip(axs, models):
            R, chi_R = EXAFS_compound(compound, simulation_type, model)
            _, chi_baseline = EXAFS_compound(compound, simulation_type, base_line_model)
            cmap = "jet"
            colors = plt.get_cmap(cmap)(np.linspace(0, 1, chi_R.shape[1]))
            ang = np.linspace(0, 2 * np.pi, chi_R.shape[0])

            for i, chi in enumerate(chi_R.T):
                # for i, chi in enumerate(chi_R.T - chi_baseline.T):
                ax.scatter(
                    ang,
                    np.abs(chi),
                    s=1,
                    alpha=0.2,
                    color=colors[i],
                )

            # if model != base_line_model:
            if True:

                # axs[-1].plot(
                #     ang,
                #     np.abs(chi_baseline.mean(axis=1)),
                #     color=model_colors[model],
                #     label=model,
                # )

                # plot moving average instead of mean to smooth the curve
                window = 5
                chi_mean = np.abs(chi_R.mean(axis=1))
                chi_mean = np.convolve(chi_mean, np.ones(window), "valid") / window
                axs[-1].plot(
                    ang[window // 2 : -window // 2 + 1],
                    chi_mean,
                    color=model_colors[model],
                    label=model,
                )

            ax.set_yscale("log")
            ax.set_rlim(None, 30)
            ax.set_xticks([])
            ax.set_yticks([])
        axs[-1].legend()

    for ax, model in zip(ax_all[0], models):
        ax.set_ylabel(model, rotation=0, labelpad=20)
        ax.yaxis.set_label_position("right")

    plt.savefig("radial_exafs.png", bbox_inches="tight", dpi=300)

    # %%

    # %%
