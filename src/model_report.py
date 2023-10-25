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
    for compound in ["Cu-O", "Ti-O"]:
        for simulation_type in ["FEFF", "VASP"]:
            if simulation_type == "VASP" and compound == "Cu-O":
                continue
            model_report(
                query={
                    "compound": compound,
                    "simulation_type": simulation_type,
                    "split": "material",
                },
                model_fn=linear_model_predictions,
            )
