# %%
# %load_ext autoreload
# %autoreload 2
from loguru import logger
import numpy as np
from typing import Tuple
from _legacy.data.ml_data import XASPlData
import glob
import os

import ast
import re
import torch
from utils.src.optuna.dynamic_fc import PlDynamicFC
import yaml


def get_optimal_fc_predictions(
    query,
    base_dir="results/oct_26/optimal_nn_tb_logs/",
):
    optimal_fc = yaml.safe_load(open("cfg/misc.yaml", "r"))["optimal_fc_params"]
    data_module = XASPlData(query=query, batch_size=128, num_workers=0)
    model_name, data, predictions = fc_ckpt_predictions(
        query=query,
        data_module=data_module,
        widths=optimal_fc[query["compound"]][query["simulation_type"]],
        base_dir=base_dir,
    )
    return model_name, data, predictions


def extract_widths_from_ckpt_path(checkpoint_path):
    match = re.search(r"(\[.*\])", checkpoint_path)
    if match:
        extracted_str = match.group(1)
        widths = ast.literal_eval(extracted_str)
        return widths
    else:
        raise ValueError("No widths found in checkpoint path")


def load_model_from_ckpt(checkpoint_path, widths):
    # widths = extract_widths_from_ckpt_path(checkpoint_path)
    model = PlDynamicFC(widths=widths, output_size=200)
    checkpoint = torch.load(checkpoint_path)
    corrected_state_dict = {
        k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(corrected_state_dict)
    model.eval()
    return model


def find_ckpt_paths(query, base_dir):
    ckpt_path = ""
    compound = query.get("compound", "")
    sim_type = query.get("simulation_type", "").lower()
    pattern = os.path.join(base_dir, f"{compound}-{sim_type}/**/checkpoints/*.ckpt")
    matching_files = glob.glob(pattern, recursive=True)
    if matching_files:
        ckpt_path = matching_files[0]  # Assuming one match
    return ckpt_path


def fc_ckpt_predictions(
    query,
    data_module,
    widths,
    base_dir="results/oct_26/optimal_nn_tb_logs/",
) -> Tuple[str, np.ndarray, np.ndarray]:
    ckpt_path = find_ckpt_paths(query, base_dir)

    if not ckpt_path:
        logger.warning(f"No checkpoint found for query {query}")
        return str(), np.ndarray([]), np.ndarray([])

    # Load the model
    model = load_model_from_ckpt(ckpt_path, widths)

    predictions = torch.tensor([])
    data = torch.tensor([])
    for batch in data_module.test_dataloader():
        x, y = batch
        data = torch.cat((data, y), dim=0)
        predictions = torch.cat((predictions, model(x)), dim=0)

    predictions = predictions.detach().numpy()
    data = data.detach().numpy()
    return f"fc_{model.widths}", data, predictions


if __name__ == "__main__":
    base_dir = "results/oct_25/optimal_nn_tb_logs/"
    query = {
        "compound": "Cu-O",
        "simulation_type": "FEFF",
        "split": "material",
    }

    model_name, data, predictions = get_optimal_fc_predictions(query=query)
    print(f"Model name: {model_name}")
    print(f"Data shape: {data.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # optimal_widths = {
    #     "Cu-O": {
    #         "FEFF": [64, 180, 200],
    #     },
    #     "Fe-O": {
    #         "VASP": [64, 190, 180],
    #         "FEFF": [64, 150, 120, 170],
    #     },
    # }
    # data_module = XASPlData(query=query, batch_size=128, num_workers=0)
    # model_name, data, predictions = fc_ckpt_predictions(
    #     query,
    #     base_dir,
    #     data_module,
    #     widths=optimal_widths[query["compound"]][query["simulation_type"]],
    # )


# %%
