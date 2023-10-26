from src.ckpt_predictions import fc_ckpt_predictions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from scripts.plots_model_report import generate_plots_for_report
from src.xas_data import XASData


def model_report(query, model_fn):
    model_name, data, predictions = model_fn(query=query)

    # texts
    compound = query["compound"]
    simulation_type = query["simulation_type"]
    save_file_prefix = f"{model_name}_{compound}_{simulation_type}"
    title_prefix = save_file_prefix + "\n"

    # plots for data, predictions, residues ..
    generate_plots_for_report(
        data=data,
        predictions=predictions,
        save=False,
        title_prefix=title_prefix,
        save_file_prefix=save_file_prefix,
    )


def linear_model_predictions(query):
    data = XASData.load_data(query=query)
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]

    model_name = "linear_regression"
    predictions = LinearRegression().fit(X_train, y_train).predict(X_test)
    return model_name, y_test, predictions


if __name__ == "__main__":
    base_dir = "results/oct_25/optimal_nn_tb_logs/"
    optimal_widths = {
        "Cu-O": {
            "FEFF": [64, 180, 200],
        },
        "Fe-O": {
            "VASP": [64, 190, 180],
            "FEFF": [64, 150, 120, 170],
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
            model_report(
                query=query,
                model_fn=linear_model_predictions,
            )

    data_module = XASData(query=query, batch_size=128, num_workers=0)
    model_name, data, predictions = fc_ckpt_predictions(
        query,
        base_dir,
        data_module,
        widths=optimal_widths[query["compound"]][query["simulation_type"]],
    )

# %%
