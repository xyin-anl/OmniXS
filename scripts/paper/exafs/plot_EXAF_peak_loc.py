from EXAFS import EXAFS_compound
from config.defaults import cfg


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def generate_kde_plot(compound, simulation_type, model, ax=None, method="max"):
    FONTSIZE = 18
    chi_R = EXAFS_compound(compound, simulation_type, model)[1]
    # fn = np.angle
    fn = np.abs
    # fn = np.real
    if method == "max":
        kde = gaussian_kde(fn(np.max(chi_R, axis=1)))
    if method == "max_loc":
        kde = gaussian_kde(fn(np.argmax(chi_R, axis=1)))
    if method == "first_peak":
        kde = gaussian_kde(fn(np.argmax(np.gradient(chi_R, axis=1), axis=1)))
    if method == "first_peak_loc":
        kde = gaussian_kde(fn(np.max(np.gradient(chi_R, axis=1), axis=1)))

    x = np.linspace(-10, 20, 100)

    # put background color
    cmap = "tab10"
    feff_colors = {c: plt.get_cmap(cmap)(i) for i, c in enumerate(cfg.compounds)}
    vasp_colors = {
        "Cu": plt.get_cmap(cmap)(len(feff_colors) + 0),
        "Ti": plt.get_cmap(cmap)(len(feff_colors) + 1),
    }
    ax.text(
        0.8,
        0.9,
        compound if simulation_type == "FEFF" else f"{compound}_VASP",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        bbox=dict(
            facecolor=(
                feff_colors[compound]
                if simulation_type == "FEFF"
                else vasp_colors[compound]
            ),
            alpha=0.3,
        ),
        fontsize=FONTSIZE * 0.8,
    )

    ax.plot(
        x,
        kde(x),
        label=f"{model}",
    )
    ax.set_xlim(0, 20)


# %%
method = "first_peak_loc"

# # FEFF
simulation_type = "FEFF"
compounds = cfg.compounds
models = ["simulation", "expert", "universal", "tuned_universal"]

# # VASP
# simulation_type = "VASP"
# compounds = ["Cu", "Ti"]
# models = ["simulation", "expert", "tuned_universal"]


fig = plt.figure(figsize=(8, 2 * (len(compounds) // 2)))
gs = fig.add_gridspec(len(compounds) // 2, 2, hspace=0, wspace=0)
axs = gs.subplots(sharex=True, sharey=False).reshape(1, -1)

for ax, compound in zip(axs.flatten(), compounds):
    plt.style.use(["default", "science"])
    for model in models:
        generate_kde_plot(
            compound,
            simulation_type,
            model,
            ax=ax,
            method=method,
        )

FONTSIZE = 18
for ax in axs[-1, :]:
    xticks = np.linspace(ax.get_xticks()[0], ax.get_xticks()[-1], 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in ax.get_xticks()], fontsize=FONTSIZE * 0.8)
    ax.set_xlabel(r"$R_{\max}$", fontsize=FONTSIZE)
for ax in axs[:, 0]:
    yticks = np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()], fontsize=FONTSIZE * 0.8)
    ax.set_ylim(0, None)
    ax.set_ylabel("KDE", fontsize=FONTSIZE)
axs[0][0].legend(fontsize=FONTSIZE * 0.8)

fig.tight_layout()
fig.savefig(f"first_peak_loc_{simulation_type}.pdf", bbox_inches="tight", dpi=300)
