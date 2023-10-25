import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression

from src.xas_data import XASData
from utils.src.plots.heatmap_of_lines import heatmap_of_lines


def plot_residue_quartiles(residues, offset, compound, simulation_type):
    for percentile in np.arange(10, 100, 10):
        y_percentile = np.percentile(residues, percentile, axis=0)
        plt.plot(
            offset + y_percentile,
            alpha=0.5,
            linestyle="--",
            label=f"percentile: {percentile}%",
        )
    plt.title(f"Model consistency: {compound} {simulation_type}")
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("spectra_mean + residue_percentile")
    plt.tight_layout()


def plot_residue_histogram(residues, title):
    plt.hist(residues, bins=200, density=True)
    plt.title(title)
    plt.xlabel("residue")
    plt.ylabel("probabilty density")


def plot_residue_cv(residues, title):
    cv_values = np.abs(residues).std(axis=0) / np.abs(residues).mean(axis=0)
    plt.plot(cv_values, marker=".", linestyle="-")
    plt.xlabel("index")
    plt.ylabel("Coefficient of Variation of residues")
    plt.title(title)
    plt.ylim([0, max(cv_values) * 1.1])


def plot_residue_heatmap(residues, title):
    linreg_rmse = np.sqrt(np.mean((((residues) ** 2)).flatten()))
    heatmap_of_lines(residues, ylabel="residues")
    plot_title = title
    plot_title += f"\n RMSE = {round(linreg_rmse, 3)}"
    plt.title(plot_title)


# def save_plot(file_name):
#     from copy import deepcopy
#     current_ax = plt.gca()
#     temp_fig = plt.figure()
#     temp_ax = deepcopy(current_ax)
#     temp_fig.add_axes(temp_ax)
#     temp_fig.savefig(file_name, dpi=300)
#     plt.close(temp_fig)


def save_plot(file_name):
    raise NotImplementedError


def plot_data(data, title_prefix):
    heatmap_of_lines(data, ylabel="spectral data")
    plt.title(f"{title_prefix}")


def plot_predictions(predictions, residues, title_prefix):
    heatmap_of_lines(predictions, ylabel="spectral predictions")
    linreg_rmse = np.sqrt(np.mean((((residues) ** 2)).flatten()))
    plt.title(f"{title_prefix}:  RMSE = {round(linreg_rmse, 3)}")


def generate_plots_for_report(
    predictions,
    data,
    title_prefix,
    save_file_prefix,
    save: bool = False,
):
    residues = predictions - data

    # plot settings
    plt.style.use(["science", "notebook", "high-vis", "no-latex"])  # TODO
    width = 15
    fig = plt.figure(figsize=(width, 0.618 * 4 * width))
    gs = GridSpec(5, 2, figure=fig)
    ax_data = fig.add_subplot(gs[0, :])
    ax_predictions = fig.add_subplot(gs[1, :], sharex=ax_data)
    ax_residues = fig.add_subplot(gs[2, :], sharex=ax_data)
    ax_residue_cv = fig.add_subplot(gs[3, 0], sharex=ax_data)
    ax_residues_histogram = fig.add_subplot(gs[3, 1])

    # to check if the model is consistent along indices
    plt.sca(ax_residue_cv)
    plot_residue_cv(
        residues,
        title=f"{title_prefix} Coefficient of Variation of residues",
    )
    if save:
        save_plot(f"{save_file_prefix}_residue_cv.pdf")

    # to check the symmetry of residues
    plt.sca(ax_residues_histogram)
    plot_residue_histogram(
        residues.flatten(),
        title=f"{title_prefix} Histogram of residues",
    )
    if save:
        save_plot(f"{save_file_prefix}_residue_histogram.pdf")

    plt.sca(ax_residues)
    plot_residue_heatmap(residues, title=f"{title_prefix} Residue heatmap")
    if save:
        save_plot(f"{save_file_prefix}_residue_heatmap.pdf")

    plt.sca(ax_data)
    plt.gca().set_aspect(1.618)
    plot_data(data, title_prefix=f"{title_prefix} Heatmap of test data")
    if save:
        save_plot(f"{save_file_prefix}_heatmap_test_data.pdf")

    plt.sca(ax_predictions)
    plt.gca().set_aspect(1.618)
    plot_predictions(predictions, residues, title_prefix=f"{title_prefix} Predictions")
    if save:
        save_plot(f"{save_file_prefix}_heatmap_predictions.pdf")

    plt.tight_layout()
    plt.savefig(f"{save_file_prefix}.pdf", dpi=300)


if __name__ == "__main__":
    # configs
    compound = "Cu-O"
    simulation_type = "FEFF"  # "VASP", "FEFF"
    query = {
        "compound": compound,
        "simulation_type": simulation_type,
        "split": "material",
    }

    # linear_model_report(query=query)

    # # data
    # data = XASData.load_data(query=query)
    # X_train, y_train = data["train"]["X"], data["train"]["y"]
    # X_test, y_test = data["test"]["X"], data["test"]["y"]

    # # model and predictions
    # model_name = "Linear Regression"
    # model = LinearRegression().fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # y_residue = y_test - y_pred

    # generate_plots_for_report(
    #     data=y_test,
    #     predictions=y_pred,
    #     save=False,
    #     title_prefix=f"{model_name}_{compound}_{simulation_type} \n",
    #     save_file_prefix=f"{model_name}_{compound}_{simulation_type}",
    # )
