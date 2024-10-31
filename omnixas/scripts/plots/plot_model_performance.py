# %%


import numpy as np
from matplotlib import pyplot as plt

from omnixas.data import ElementsFEFF, ElementsVASP
from omnixas.scripts.plots.scripts import AllEtas

# %%


def plot_performance_comparison(etas, figsize=(8, 6), fontsize=20):

    # reformat eta_values to be model_name -> element_type -> eta
    eta_values = {}
    for model_tag, eta in etas.items():
        model_name = model_tag.name
        element_type = f"{model_tag.element}_{model_tag.type}"
        eta_values.setdefault(model_name, {})[element_type] = eta

    plt.style.use(["default", "science", "tableau-colorblind10"])
    fig, ax = plt.subplots(figsize=figsize)

    # Constants for plotting
    BAR_CENTER_FACTOR = 1.5
    bar_width = 0.95 / 3  # 3 models

    # Use ordered elements from enums
    feff_elements = list(ElementsFEFF)
    vasp_elements = list(ElementsVASP)

    # Create color map for elements
    compound_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(feff_elements) + 2))

    # Define hatches and fill properties for each model
    model_props = {
        "universalXAS": {"hatch": "..", "fill": False, "position": 0},
        "expertXAS": {"hatch": ".....", "fill": False, "position": 1},
        "tunedUniversalXAS": {"hatch": "", "fill": True, "position": 2},
    }

    # Plot bars for each model
    for model_name, model_data in eta_values.items():
        # Plot FEFF data
        x_positions = np.arange(len(feff_elements)) * BAR_CENTER_FACTOR
        values = [model_data.get(f"{elem}_FEFF", np.nan) for elem in feff_elements]

        ax.bar(
            x_positions + model_props[model_name]["position"] * bar_width,
            values,
            bar_width,
            label=model_name,
            edgecolor=[compound_colors[i] for i in range(len(values))],
            fill=model_props[model_name]["fill"],
            color=compound_colors[: len(values)],
            hatch=model_props[model_name]["hatch"],
            zorder=3,
        )

        # Plot VASP data if available
        if any(k.endswith("_VASP") for k in model_data.keys()):
            vasp_values = [
                model_data.get(f"{elem}_VASP", np.nan) for elem in vasp_elements
            ]
            vasp_positions = (
                np.arange(len(vasp_elements)) + len(feff_elements)
            ) * BAR_CENTER_FACTOR

            ax.bar(
                vasp_positions + model_props[model_name]["position"] * bar_width,
                vasp_values,
                bar_width,
                edgecolor=compound_colors[-len(vasp_elements) :],
                fill=model_props[model_name]["fill"],
                color=compound_colors[-len(vasp_elements) :],
                hatch=model_props[model_name]["hatch"],
                zorder=3,
            )

    # Styling
    ax.set_ylabel(r"Performance ($\eta$)", fontsize=fontsize * 1.2)
    ax.set_xlabel("Element", fontsize=fontsize * 1.2, labelpad=-10)

    # Set x-ticks
    all_tick_positions = np.arange(len(feff_elements)) * BAR_CENTER_FACTOR + bar_width
    vasp_tick_positions = (
        np.arange(len(vasp_elements)) + len(feff_elements)
    ) * BAR_CENTER_FACTOR + bar_width
    all_tick_positions = np.concatenate([all_tick_positions, vasp_tick_positions])

    # Add VASP region
    ax.axvspan(
        all_tick_positions[-2] - bar_width * 1.5,
        ax.get_xlim()[1],
        alpha=0.1,
        color="grey",
    )

    # Add VASP text
    ax.text(
        all_tick_positions[-1] - bar_width / 2,
        ax.get_ylim()[1] * 0.75,
        "VASP",
        fontsize=fontsize * 1.2,
        ha="center",
        va="center",
        color="grey",
    )

    # Set tick labels
    ax.set_xticks(all_tick_positions)
    all_labels = list(feff_elements) + [
        f"{elem}\n" + r"{\Large VASP}" for elem in vasp_elements
    ]
    ax.set_xticklabels(all_labels, fontsize=fontsize)

    # Add legend
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, hatch=props["hatch"], fill=props["fill"], color="gray"
        )
        for props in model_props.values()
    ]

    legend_labels = {
        "universalXAS": "UniversalXAS",
        "expertXAS": "ExpertXAS",
        "tunedUniversalXAS": "Tuned-UniversalXAS",
    }

    ax.legend(
        handles,
        [legend_labels[model] for model in model_props.keys()],
        fontsize=fontsize * 0.8,
        handlelength=2,
        handleheight=1,
        loc=(0.41, 0.75),
    )

    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()

    return fig, ax


# %%

if __name__ == "__main__":

    from omnixas.data import ThousandScaler
    from omnixas.utils import DEFAULTFILEHANDLER

    fig, ax = plot_performance_comparison(
        AllEtas(
            x_scaler=ThousandScaler,
            y_scaler=ThousandScaler,
            file_handler=DEFAULTFILEHANDLER(),
        )
    )
    plt.savefig("model_performance.pdf", bbox_inches="tight", dpi=300)

    # %%
