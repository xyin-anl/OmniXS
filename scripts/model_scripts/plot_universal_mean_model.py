from matplotlib import pyplot as plt

from config.defaults import cfg
from src.data.ml_data import DataQuery
from src.models.trained_models import MeanModel
import scienceplots


def plot_universal_mean_model_mse():
    mean_mse = {c: MeanModel(DataQuery(c, "FEFF")).mse for c in cfg.compounds}
    mean_mse["ALL"] = MeanModel(DataQuery("ALL", "FEFF")).mse
    plt.style.use(["default", "science"])
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(mean_mse.keys(), mean_mse.values(), edgecolor="black")
    ax.set_ylabel("MSE of Universal Mean Model", fontsize=20)
    ax.set_xlabel("Compound", fontsize=20)
    ax.set_title("MSE of Universal Mean Models", fontsize=24)
    ax.set_xticklabels(mean_mse.keys(), fontsize=18)
    plt.savefig("unviersal_mean_models_mse.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    plot_universal_mean_model_mse()
    plt.show()
