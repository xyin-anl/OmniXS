import pandas as pd
import seaborn as sns
from src.pl_data import XASData
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import scienceplots


def plot_pcas(max_pca_dim, x_pca, y_pca):
    plt.figure(figsize=(20, 20))  # Increase figure size
    for i in range(max_pca_dim):
        for j in range(i + 1, max_pca_dim):
            plt.subplot(max_pca_dim, max_pca_dim, i * max_pca_dim + j + 1)
            plt.scatter(x_pca[:, i], x_pca[:, j], c=y_pca, cmap="viridis", marker=".")
            plt.title(f"x_pca_dim = {i}, {j}")
    plt.tight_layout()


def linear_fit_of_pcas(x_pca, y_pca, selected_x_pca_dim, compound, simulation_type):
    xy_pca = np.stack([x_pca[:, selected_x_pca_dim], y_pca[:, 0]], axis=1)
    xy_pca_sorted = np.sort(xy_pca, axis=0)

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

    m = LinearRegression().fit(xy_pca_sorted[:, 0].reshape(-1, 1), xy_pca_sorted[:, 1])
    slope = m.coef_[0]
    intercept = m.intercept_
    r2 = m.score(xy_pca_sorted[:, 0].reshape(-1, 1), xy_pca_sorted[:, 1])

    ax1.plot(
        xy_pca_sorted[:, 0],
        xy_pca_sorted[:, 0] * slope + intercept,
        c="red",
        label=f"R^2 = {round(r2,2)}",
    )
    ax1.legend()
    ax1.scatter(
        xy_pca_sorted[:, 0],
        xy_pca_sorted[:, 1],
        marker=".",
        facecolors="none",
        edgecolors="blue",
        alpha=0.5,
    )
    ax1.set_xlabel(f"x_pca_dim = {selected_x_pca_dim}")
    ax1.set_ylabel("y_pca_dim = 0")

    ax2.scatter(
        x_pca[:, selected_x_pca_dim],
        x_pca[:, 0],
        c=y_pca,
        marker=".",
        cmap="viridis",
    )
    ax2.set_ylabel("x_pca_dim = 0")
    ax2.set_xticklabels([])  # This line hides the x-axis labels for ax2

    title = f"{compound} {simulation_type}: "
    title += "Linear fit of two PCA dimensions"
    ax2.set_title(title)


if __name__ == "__main__":
    # configs
    compound = "Cu-O"
    simulation_type = "FEFF"  # "VASP", "FEFF"
    query = {
        "compound": compound,
        "simulation_type": simulation_type,
        "split": "material",
    }

    plt.style.use(["science", "high-vis", "no-latex"])
    selected_x_pca_dims = {"FEFF": {"Cu-O": 2, "Ti-O": 4}, "VASP": {"Ti-O": 2}}
    selected_x_pca_dim = selected_x_pca_dims[simulation_type][compound]

    # data
    data = XASData.load_data(query=query)
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]

    # model and predictions
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_residue = y_test - y_pred

    # pca
    max_dim = 5
    x_pca = PCA(n_components=max_dim).fit_transform(X_train)
    y_pca = PCA(n_components=1).fit_transform(y_train)
    plot_pcas(max_dim, x_pca, y_pca)
    plt.savefig(f"{compound}_{simulation_type}_pcas.pdf", dpi=300)
    linear_fit_of_pcas(
        x_pca,
        y_pca,
        selected_x_pca_dim,
        compound=compound,
        simulation_type=simulation_type,
    )
    plt.savefig(f"{compound}_{simulation_type}_linear_fit_of_pcas.pdf", dpi=300)
