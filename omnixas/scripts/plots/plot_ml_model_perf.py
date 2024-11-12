# %%


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

# %%


def plot_model_performance(df):
    # Define plot parameters

    PLOT_MODELS = ["Linear", "XGB", "MLP"]
    labels = [
        "Linear Regression",
        "Gradient Boosted Regression",
        "Multi-layer Perceptron",
    ]

    # PLOT_MODELS = ["MLP", "Linear", "XGB"]
    # labels = ["MLP", "Linear Regression", "XGBoost"]

    WIDTH = 0.15
    FONTSIZE = 24

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.style.use(["default", "science"])

    def lighten_color(color, amount=0.5):
        try:
            c = mcolors.cnames[color]
        except Exception:
            c = color
        c = mcolors.to_rgba(c)
        c = [1 - (1 - component) * amount for component in c]
        return c

    # Setup colors
    model_colors = {
        k: v
        for k, v in zip(PLOT_MODELS, plt.get_cmap("tab10")(range(len(PLOT_MODELS))))
    }

    # Prepare data for plotting
    features = ["ACSF", "SOAP", "Transfer"]

    # Plot boxplots for each feature and model combination
    for i, feature in enumerate(features):
        for j, model_type in enumerate(PLOT_MODELS):
            # Get data for current feature and model type
            if feature == "Transfer":
                model_name = f"Transfer {model_type}"
            else:
                model_name = f"{feature} {model_type}"

            data = df[model_name].values

            # Create boxplot
            ax.boxplot(
                data,
                positions=[i + (j - 1) * WIDTH],
                widths=WIDTH,
                patch_artist=True,
                boxprops=dict(
                    facecolor=lighten_color(model_colors[model_type], amount=0.5)
                ),
                capprops=dict(color="black"),
                whiskerprops=dict(color="gray"),
                flierprops=dict(markeredgecolor="gray"),
                medianprops=dict(color="black"),
                showfliers=True,
            )

    # Customize plot
    ax.set_yscale("log")
    ax.set_ylim(1, 20)
    ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.5)

    # Set yticks
    yticks = np.arange(1, ax.get_ylim()[1], 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(x)}" for x in ax.get_yticks()], fontsize=FONTSIZE * 0.8)

    # Set xticks
    ax.xaxis.set_ticks_position("none")
    ax.set_xticks([0, 0.95, 1.9])
    ax.set_xticklabels(["ACSF", "SOAP", "Transfer-feature"], fontsize=FONTSIZE * 0.8)
    ax.set_xlim(-2 * WIDTH, 2 + (len(PLOT_MODELS) - 1) * WIDTH)

    # Add legend
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fc=lighten_color(model_colors[model], amount=0.5),
            edgecolor="black",
        )
        for model in PLOT_MODELS
    ]
    ax.legend(handles, labels, fontsize=FONTSIZE * 0.8, loc="upper left")

    # Labels
    ax.set_ylabel(r"Performance ($\eta$)", fontsize=FONTSIZE, labelpad=3)
    ax.set_xlabel("Features", fontsize=FONTSIZE, labelpad=10)

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    df = pd.read_json("dataset/misc/dscrb_etas.json")
    fig, ax = plot_model_performance(df)
    plt.savefig(
        "model_performance_by_feature_boxplot.pdf", bbox_inches="tight", dpi=300
    )
