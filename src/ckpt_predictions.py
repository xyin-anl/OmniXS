# %%
# %load_ext autoreload
# %autoreload 2
from src.xas_data import XASData
import glob
import os

import ast
import re
import scienceplots
import torch
from utils.src.optuna.dynamic_fc import PlDynamicFC


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


def fc_ckpt_predictions(query, base_dir, data_module, widths):
    ckpt_path = find_ckpt_paths(query, base_dir)

    if not ckpt_path:
        print(f"No checkpoint found for query {query}")
        return None, None, None

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
    optimal_widths = {
        "Cu-O": {
            "FEFF": [64, 180, 200],
        },
        "Fe-O": {
            "VASP": [64, 190, 180],
            "FEFF": [64, 150, 120, 170],
        },
    }
    data_module = XASData(query=query, batch_size=128, num_workers=0)
    model_name, data, predictions = fc_ckpt_predictions(
        query,
        base_dir,
        data_module,
        widths=optimal_widths[query["compound"]][query["simulation_type"]],
    )

# %%
