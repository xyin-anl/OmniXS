import json
import os
from typing import Dict, List

from omnixas.data import MaterialSplitter, MLData, MLSplits


def fetch_dataset_elements(data_dir="dataset/omnixas_2/features/m3gnet/"):
    files = os.listdir(data_dir)
    elements = [file.split("_")[-1].split(".")[0] for file in files if "json" in file]
    return elements


def make_ml_split(
    spectra: Dict[str, Dict[str, List[float]]],
    features: Dict[str, Dict[str, List[float]]],
    target_fractions=[0.8, 0.1, 0.1],
    seed=42,
) -> MLSplits:
    idSite = [(id, site) for id in spectra.keys() for site in spectra[id].keys()]
    split_idSite = MaterialSplitter.split(
        idSite=idSite,
        target_fractions=target_fractions,
        seed=seed,
    )
    train_idSite, val_idSite, test_idSite = split_idSite

    def to_ml_data(idSite):
        return MLData(
            X=[features[id][site] for id, site in idSite],
            y=[spectra[id][site] for id, site in idSite],
        )

    train_data = to_ml_data(train_idSite)
    val_data = to_ml_data(val_idSite)
    test_data = to_ml_data(test_idSite)
    split = MLSplits(train=train_data, val=val_data, test=test_data)
    return split


data_dir = "dataset/omnixas_2"
elements = fetch_dataset_elements()
for element in elements:
    print(element)
    spectra_dict = json.load(open(f"{data_dir}/spectra/spectra_{element}.json"))
    feature_dict = json.load(
        open(f"{data_dir}/features/m3gnet/feature_m3gnet_{element}.json")
    )
    split = make_ml_split(
        spectra_dict,
        feature_dict,
        target_fractions=[0.8, 0.1, 0.1],
    )
    with open(f"{data_dir}/splits/split_{element}.json", "w") as f:
        f.write(split.json())
