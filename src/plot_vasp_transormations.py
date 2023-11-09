import matplotlib.pyplot as plt

from src.vasp_data_transformations import VASPDataModifier


class VASPDataTransformationPlotter(VASPDataModifier):
    def __init__(self, spectra_params, id, simulation_type):
        super().__init__(spectra_params)
        self.id = id
        self.simulation_type = simulation_type
        self.compound = self.id[0]

    def legend_truncation(self):
        l_trunc = r"Truncation\\ "
        l_trunc += r"$E_{\text{min}} = e_{\text{cbm}} - e_{\text{core}} - \Delta E_0$\\"
        l_trunc += r"$E_{\text{min}} ="
        l_trunc += f"{self.e_cbm:.2f} - {self.e_core:.2f} - {self.start_offset} "
        l_trunc += r"\text{eV}$\\ "
        l_trunc += r"$E_{\text{min}} = "
        l_trunc += f"{self.energy[0]:.2f}" + r"\ \text{eV}$" + " (available)\n"
        l_trunc += r"$E_{\text{max}} = " + f"{self.energy[-1]:.2f}" + r"\ \text{eV}$"
        return l_trunc

    def legend_scaling(self):
        l_scaling = r"Scaling: \\ "
        l_scaling += r"$\omega' = \omega \cdot \Omega / \alpha$ \\ "
        l_scaling += r"$\omega = y \cdot E$ \\ "
        l_scaling += f"${{\\Omega}}= {self.big_omega:.2f}$ \n "
        l_scaling += f"${{\\alpha}}= {self.alpha:.5f}$"
        return l_scaling

    def legend_broadening(self):
        l_broadening = r"Lorentzian Broadening: \\ "
        l_broadening += r"PDF of Cauchy distribution \\ "
        l_broadening += r"$f(x, k) = \frac{1}{2^{k/2-1} \gamma \left( k/2 \right)}$"
        l_broadening += r"$x^{k-1} \exp \left( -x^2/2 \right)$ \\ "
        l_broadening += r"$\gamma = \Gamma / 2$ \\ "
        l_broadening += r"$\Gamma = 0.89$"
        return l_broadening

    def legend_alignment(self):
        l_align = r"Alignment: \\ "
        l_align += r"$\Delta E = (\epsilon_{\text{core}} - \epsilon_{\text{cbm}})$"
        l_align += r"$+ (E_{\text{ch}} - E_{\text{GS}})$ \\ "
        l_align += f"$\\Delta E = ({self.e_core:.2f} - {self.e_cbm:.2f}) "
        l_align += f"({self.E_ch:.2f} - {self.E_GS:.2f})$ \n "
        l_align += r"$\Delta E = $" f"{self.align_offset:.2f} \n"
        return l_align

    def plot(self):
        fig, ax = plt.subplots(6, 1, figsize=(15, 15))
        ax[0].plot(self.energy_full, self.spectra_full, label="Raw")
        ax[1].plot(
            self.energy_trunc,
            self.spectra_trunc,
            label=self.legend_truncation(),
        )
        ax[2].plot(
            self.energy_trunc,
            self.spectra_scaled,
            label=self.legend_scaling(),
        )
        ax[3].plot(
            self.energy_trunc,
            self.broadened_amplitude,
            label=self.legend_broadening(),
        )
        ax[4].plot(
            self.energy_aligned,
            self.broadened_amplitude,
            label=self.legend_alignment(),
        )
        ax[5].plot(
            self.energy_aligned,
            self.broadened_amplitude,
            label="brodened",
            linewidth=2,
        )
        ax[5].plot(
            self.energy_aligned,
            self.spectra_scaled,
            label="no_brodening",
            alpha=0.5,
        )

        for axis in ax:
            axis.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=13,
            )
            axis.set_xlabel("Energy (eV)", fontsize=15)
            y_label = r"$\epsilon$" if axis in ax[:2] else r"$\sigma (E)$"
            axis.set_ylabel(y_label, fontsize=15)
        fig_title = "data_transformations"
        fig_title += f"_{self.simulation_type}_{self.compound}_{self.id}"
        fig.suptitle(fig_title, fontsize=18, color="red")
        plt.tight_layout()
        plt.savefig(f"{fig_title}.pdf", dpi=300)


if __name__ == "__main__":
    from src.raw_data_vasp import RAWDataVASP

    compound = "Ti"
    simulation_type = "VASP"
    data = RAWDataVASP(compound, simulation_type)

    # id = next(iter(data.parameters))
    id = ("mp-390", "000_Ti")  # reference to another paper data

    # # %%
    import scienceplots
    plt.style.use(["science", "vibrant"])
    plotter = VASPDataTransformationPlotter(
        data.parameters[id], compound, simulation_type
    )
    plotter.plot()
    plt.show()
