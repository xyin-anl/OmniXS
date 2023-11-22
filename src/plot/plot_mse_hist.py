import matplotlib.pyplot as plt
import numpy as np

from utils.src.plots.highlight_tick import highlight_tick


def plot_mse_histogram(
    mse,
    title=None,
    ax=None,
    plot_probability=True,
    xticks_precision=3,
):
    if ax is None:
        ax = plt.gca()

    counts, bins = np.histogram(mse, bins=50)
    if plot_probability:
        counts = counts / counts.sum()
        counts = np.round(counts * 100, 3)
    # histogram
    ax.bar(
        bins[:-1],
        counts,
        width=bins[1] - bins[0],
        edgecolor="black",
    )
    ax.set_xticks(bins)
    highlight_tick(
        ax=ax,
        highlight=mse.mean(),
        axis="x",
        precision=xticks_precision,
    )
    ax.axvline(
        mse.mean(),
        color="r",
        linestyle="--",
        label=f"mean MSE: {mse.mean():.3f}",
    )
    ax.legend()

    # limits the x axis to the 98th percentile to avoid outliers
    x_lim_min = bins[0] - (bins[1] - bins[0])
    limit_percentile = 0.98
    x_lim_max = np.quantile(mse, limit_percentile)
    ax.set_xlim(x_lim_min, x_lim_max)

    ax.set_title(title)
    ax.set_xlabel(f"MSE (to {limit_percentile * 100:.0f}th percentile)")
    ax.set_ylabel("percentages (%)" if plot_probability else "counts")
    ax.figure.tight_layout()
    return ax


if __name__ == "__main__":
    from src.ckpt_predictions import get_optimal_fc_predictions

    for coumound in ["Cu-O", "Ti-O"]:
        for simulation_type in ["VASP", "FEFF"]:
            if coumound == "Cu-O" and simulation_type == "VASP":
                continue
            query = {
                "compound": coumound,
                "simulation_type": simulation_type,
                "split": "material",
            }
            model_name, data, predictions = get_optimal_fc_predictions(query=query)
            # model_name, data, predictions = linear_model_predictions(query=query)
            with plt.style.context(["nature", "no-latex"]):
                plot_mse_histogram(
                    mse=np.power(predictions - data, 2).mean(axis=1),
                    title=f"{query['compound']}_{query['simulation_type']} {model_name} {query['split']}_split MSE histogram",
                    ax=plt.figure(figsize=(6, 4)).gca(),
                )
                # plt.show()
                plt.savefig(
                    f"{query['compound']}_{query['simulation_type']}_{model_name}_{query['split']}_split_mse_histogram.pdf",
                    dpi=300,
                )
