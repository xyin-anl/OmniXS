import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from scripts.model_scripts.model_report import linear_model_predictions
from src.data.ml_data import XASPlData
from src.models.ckpt_predictions import fc_ckpt_predictions


def plot_best_worst_residues(query, y_test, predictions, model_name, count=10):
    y_residues = np.abs(y_test - predictions)
    df_residues = pd.DataFrame(y_residues.mean(axis=1), columns=["residues"])
    df_test_data = pd.DataFrame(y_test)
    df_predictions = pd.DataFrame(predictions)
    df_all = pd.concat([df_residues, df_test_data, df_predictions], axis=1)

    column_names = (
        ["residues"]
        + [f"test_data_{i}" for i in range(y_test.shape[1])]
        + [f"predictions_{i}" for i in range(y_test.shape[1])]
    )

    df_all.columns = column_names
    df_all = df_all.sort_values(by="residues")

    df_data = df_all[[col for col in column_names if "data" in col]]
    df_predictions = df_all[[col for col in column_names if "predictions" in col]]

    plt.style.use(["science", "notebook", "high-vis", "no-latex"])
    fig, axes = plt.subplots(count, 2, figsize=(20, 20), sharex=True, sharey=True)

    for i in range(count):
        # Plot best on the left column
        df_data.iloc[i].plot(ax=axes[i, 0], label="data")
        df_predictions.iloc[i].plot(ax=axes[i, 0], label="predictions", linestyle="--")
        axes[i, 0].set_title(f"MAE: {round(df_all.iloc[i]['residues'], 3)}")

        # Plot worst on the right column
        idx = len(df_all) - 1 - i
        df_data.iloc[idx].plot(ax=axes[i, 1], label="data")
        df_predictions.iloc[idx].plot(
            ax=axes[i, 1], label="predictions", linestyle="--"
        )
        axes[i, 1].set_title(f"MAE: {round(df_all.iloc[idx]['residues'], 3)}")

        # Remove ticks and labels
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_yticklabels([])
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_yticklabels([])

    # Add common labels and titles
    plt.suptitle(
        f"{query['compound']}_{query['simulation_type']}_{model_name}",
        fontsize=40,
    )
    fig.text(0.25, 0.98, "Best Residues", ha="center", fontsize=30, c="green")
    fig.text(0.75, 0.98, "Worst Residues", ha="center", fontsize=30, c="red")
    fig.text(0.5, 0.04, "Common X-axis", ha="center")
    fig.text(0.04, 0.5, "Common Y-axis", va="center", rotation="vertical")

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    plt.legend()
    plt.savefig(
        f"{query['compound']}_{query['simulation_type']}_{model_name}_top_residues.pdf"
    )


if __name__ == "__main__":
    base_dir = "results/oct_25/optimal_nn_tb_logs/"
    optimal_widths = {
        "Cu-O": {
            "FEFF": [64, 180, 200],
        },
        "Ti-O": {
            "FEFF": [64, 190, 180],
            "VASP": [64, 150, 120, 170],
        },
    }

    for compound in ["Cu-O", "Ti-O"]:
        for simulation_type in ["FEFF", "VASP"]:
            if simulation_type == "VASP" and compound == "Cu-O":
                continue
            query = {
                "compound": compound,
                "simulation_type": simulation_type,
                "split": "material",
            }

            # FC
            data_module = XASPlData(query=query, batch_size=1, num_workers=0)
            _, y_test, predictions = fc_ckpt_predictions(
                query=query,
                base_dir=base_dir,
                data_module=data_module,
                widths=optimal_widths[query["compound"]][query["simulation_type"]],
            )
            plot_best_worst_residues(query, y_test, predictions, model_name="FC")

            # linear
            _, y_test, predictions = linear_model_predictions(query=query)
            plot_best_worst_residues(query, y_test, predictions, model_name="LR")
