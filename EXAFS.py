# %%

#  steps:
#  find mu0(E) - background absorption
#  find chi(E) = (mu(E) - mu0(E))/mu0(E)
#  find E0 - edge energy, max derivative of mu(E), ...
#  find k - wavenumber = sqrt(2m(E-E0)/hbar) ~ sqrt(E-E0 + delta) (delta makes k real)
#  find common k grid
#  find chi(k)
#  multiply by k^2 to get EXAFS signal for visualization
#  use window function to avoid leakage (Tukey window)
import numpy as np
from scipy.fft import fft
from scipy.interpolate import UnivariateSpline, interp1d
import scienceplots

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data


class EXAFS:
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

        # return self.E[int(0.5 * len(self.E))] # arbitrary

        """Determine the absorption edge energy E0 using first and second derivatives."""
        d_mu_dE = np.gradient(self.mu_E, self.E)
        E0_index = np.argmax(d_mu_dE)
        # return self.mu_E[E0_index]

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
        return np.sqrt(self.dE) / 10

    @property
    def chi_k(self):
        idx = self.E > self.E0
        k = np.sqrt(self.E[idx] - self.E0)

        # BASED on E grid
        # k_grid = np.arange(k[0], k[-1], self.dk)

        # COMMON k grid
        K_COUNT = 100
        k_grid = np.linspace(k[0], k[-1], K_COUNT)
        interp = interp1d(k, self.X_E[idx], kind="cubic")

        return k_grid, interp(k_grid)

    @property
    def chi_k_windowed(self):
        """Apply a window function to chi(k)."""
        k_grid, chi_k = self.chi_k
        window = np.hanning(len(chi_k))  # Apply a Hanning window
        chi_k_windowed = chi_k * window
        return k_grid, chi_k_windowed

    @property
    def chi_k_fft(self):
        """Compute the Fourier Transform of the windowed chi(k) to obtain chi(R)."""
        k_grid, chi_k_windowed = self.chi_k_windowed

        chi_k_fft = fft(chi_k_windowed)
        R = np.fft.fftfreq(
            len(chi_k_windowed), d=(k_grid[1] - k_grid[0])
        )  # Compute the R grid
        R = np.fft.fftshift(R)
        chi_k_fft = np.fft.fftshift(chi_k_fft)
        chi_k_fft = chi_k_fft[R >= 0]
        R = R[R >= 0]
        return R, chi_k_fft

    def _load_E(self):
        # data = load_xas_ml_data(DataQuery(self.compound, self.simulation_type))
        data = np.load(
            cfg.paths.ml_data.format(
                compound=self.compound, simulation_type=self.simulation_type
            )
        )
        return data["energies"]

    def chi_r_peak(self):
        R, chi_R = self.chi_k_fft
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
        k, chi_k = self.chi_k
        ax[1, 0].plot(k, chi_k * k**2)
        ax[1, 0].set_xlabel(r"$k$ ($\AA^{-1}$)")
        ax[1, 0].set_ylabel(r"$\chi(k) \cdot k^2$")
        ax[1, 0].set_title("EXAFS signal")

        # plot after windowing
        k, chi_k_windowed = self.chi_k_windowed
        ax[1, 0].plot(k, chi_k_windowed * k**2)
        ax[1, 0].set_xlabel(r"$k$ ($\AA^{-1}$)")
        ax[1, 0].set_ylabel(r"$\chi(k) \cdot k^2$")
        ax[1, 0].set_title("EXAFS signal after windowing")
        ax[1, 0].legend(["Original", "Windowed"])

        R, chi_R = self.chi_k_fft
        # plot both magnitude and phase
        ax[1, 1].plot(R, np.abs(chi_R), label="Magnitude", marker="o")
        ax[1, 1].set_xlabel(r"$R$ ($\AA$)")
        ax[1, 1].set_ylabel(r"$|\chi(R)|$")
        ax[1, 1].set_title("Fourier Transform of EXAFS signal")
        ax[1, 1].set_xlim(0, 5)
        print(f"R_peak: {self.chi_r_peak()}")
        ax[1, 1].vlines(
            self.chi_r_peak()[0],
            0,
            np.abs(self.chi_r_peak()[1]),
            color="r",
            linestyle="--",
            label=r"$" + f"{self.chi_r_peak()[0]:.2f}" + r"$ $\AA$",
            alpha=0.5,
        )
        ax[1, 1].legend()

        ax1t = ax[1, 1].twinx()
        ax1t.plot(R, np.angle(chi_R), color="gray", label="Phase")
        ax1t.set_ylabel(r"$\angle \chi(R)$")
        ax1t.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # PLOT PROCESSING STEPS
    compound = "Cu"
    simulation_type = "FEFF"
    data = load_xas_ml_data(DataQuery(compound, simulation_type))
    spectras = data.test.y
    idx = np.random.randint(0, len(spectras))
    spectra = spectras[idx]
    exafs = EXAFS(spectra, compound=compound, simulation_type=simulation_type)
    exafs.plot()
