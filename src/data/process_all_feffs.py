import os
from pickle import dump, load

import numpy as np
import scienceplots
from matplotlib import pyplot as plt
from p_tqdm import p_map

from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF


def mean_filter(feff_all, drop_amount=10):
    spectras = np.array([x.spectra for x in feff_all])
    upper_bound = np.mean(spectras, axis=0) + drop_amount * np.std(spectras, axis=0)
    lower_bound = np.mean(spectras, axis=0) - drop_amount * np.std(spectras, axis=0)
    filter = np.all(((spectras <= upper_bound) & (spectras >= lower_bound)), axis=1)
    return filter


def plot_all_spectras(spectras, ax=None):
    heatmap_of_lines(np.array([x.spectra for x in spectras]), ax=ax)


def plot_filter_effect(pre_filter_feffs, post_filter_feffs):
    # Plot effect of mean filtering in spectras of each compound
    for pre, post in zip(pre_filter_feffs, post_filter_feffs):
        compound = pre[0].compound
        plt.style.use(["default", "science"])
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        plot_all_spectras(pre, ax=axs[0])
        plot_all_spectras(post, ax=axs[1])
        axs[0].set_title(f"{compound} spectras before : {len(pre)}")
        axs[1].set_title(f"{compound} spectras after mean-std filter: {len(post)}")
        plt.tight_layout()
        plt.savefig(f"mean_filter{compound}.pdf", dpi=300, bbox_inches="tight")
        plt.show()


def load_all_feff_data(compounds):
    if os.path.exists("feff_raw_all.pkl"):  # cache
        feff_raw_all = load(open("feff_raw_all.pkl", "rb"))
    else:
        feff_raw_all = p_map(lambda c: RAWDataFEFF(c), compounds)
        dump(feff_raw_all, open("feff_raw_all.pkl", "wb"))
    return feff_raw_all


def resample_all_feff_data(feff_raw_all):
    if os.path.exists("feff_all_resampled.pkl"):
        feff_all_resampled = load(open("feff_all_resampled.pkl", "rb"))
    else:
        feff_all_resampled = p_map(
            lambda d: [
                FEFFData(d.compound, d.parameters[id], id).resample() for id in d.ids
            ],
            feff_raw_all,
        )
        dump(feff_all_resampled, open("feff_all_resampled.pkl", "wb"))
    return feff_all_resampled


def filter_all_feff_data(pre_filter_feffs, std_factor):
    post_filter = {
        d[0].compound: np.array(d)[mean_filter(d, drop_amount=std_factor)]
        for d in pre_filter_feffs
    }
    return post_filter


def process_and_save_all_feffs(compounds, plot=False, STD_FACTOR=2.5):
    feff_all = load_all_feff_data(compounds)
    pre_filter_feffs = resample_all_feff_data(feff_all)
    post_filter_feff = filter_all_feff_data(pre_filter_feffs, STD_FACTOR)
    for compound, feffs in post_filter_feff.items():
        for feff in feffs:
            feff.save()
    if plot:
        plot_filter_effect(
            pre_filter_feffs, [post_filter_feff[c] for c in post_filter_feff.keys()]
        )


if __name__ == "__main__":
    compounds = ["Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Ti", "V"]
    process_and_save_all_feffs(compounds, plot=True, STD_FACTOR=2.5)

    # feff_raw_all = load_all_feff_data(compounds)
    # feff_all_resampled = p_map(
    #     lambda d: [
    #         FEFFData(d.compound, d.parameters[id], id).resample() for id in d.ids
    #     ],
    #     feff_raw_all,
    # )
