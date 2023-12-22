import os
from pickle import dump, load

import matplotlib.pyplot as plt
import numpy as np
import yaml
from p_tqdm import p_map

from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF
import scienceplots


def find_e_start_and_end(compounds, processed_data):
    start_cut_off_percentile = 0  # <---- seems good enough
    e_range = 35  # <---- inside transformations.yaml
    e_start_all = {
        compound: np.array([d.energy[0] for d in data])
        for compound, data in processed_data.items()
    }
    e_start = {
        compound: np.percentile(e_start_all[compound], start_cut_off_percentile)
        for compound in compounds
    }
    e_end = {compound: e_start[compound] + e_range for compound in compounds}

    plot_truncation_related_histograms(
        compounds,
        e_start_all,
        start_cut_off_percentile,
        e_range,
        e_start,
        e_end,
        processed_data,
    )

    return e_start, e_end


def find_peaks(processed_data):
    return {
        compound: np.array([d.energy[np.argmax(d.spectra)] for d in data])
        for compound, data in processed_data.items()
    }


def filter_off_peaks_data(compounds, processed_data, e_start, e_end):
    peaks = find_peaks(processed_data)
    capture_filter = {
        compound: (peaks[compound] < e_end[compound])
        & (peaks[compound] > e_start[compound])
        for compound in compounds
    }
    filtered_data = {
        compound: processed_data[compound][capture_filter[compound]]
        for compound in compounds
    }
    return filtered_data


def plot_truncation_related_histograms(
    compounds,
    e_start_all,
    start_cut_off_percentile,
    e_range,
    e_start,
    e_end,
    processed_data,
):
    peaks = find_peaks(processed_data)
    plt.style.use(["default", "science"])
    fig, axs = plt.subplots(4, 2, figsize=(10, 12))
    for ax, compound in zip(axs.flatten(), compounds):
        kwargs = {"bins": 50, "density": True}
        ax.hist(peaks[compound], **kwargs, label="Peaks")
        ax.hist(e_start_all[compound], **kwargs, label=r"$E_{start}$")
        kwargs = {"linestyle": "dashed", "linewidth": 2}
        ax.axvline(
            e_start[compound],
            **kwargs,
            label=r"$E_{start} = $"
            + f" {start_cut_off_percentile} percentile of "
            + r"$E_{start}$",
            color="tab:green",
        )
        ax.axvline(
            e_end[compound],
            **kwargs,
            label=r"$E_{end} = E_{start}$ + " + f"{ e_range}",
            color="tab:blue",
        )
        ax.set_xlim(e_start[compound] - 1, e_end[compound] + 1)
        ax.set_ylim(0, 1)
        ax.set_title(compound, fontsize=16)
        ax.text(
            0.5,
            0.8,
            f"{e_start[compound]:.1f} - {e_end[compound]:.1f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="tab:red", boxstyle="round"),
            color="tab:red",
        )
    for ax in axs[-1, :]:
        ax.set_xlabel("Energy (eV)", fontsize=14)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=14, loc="lower center")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.suptitle(
        "Energy range selection for all compounds",
        fontsize=22,
        fontweight="bold",
    )
    fig.savefig("energy_range_selection.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_count_of_filtered_data(compounds, processed_data, filtered_data):
    sorted_compounds = np.array(
        sorted(
            list(
                zip(
                    compounds, [len(processed_data[compound]) for compound in compounds]
                )
            ),
            key=lambda x: x[1],
            reverse=True,
        )
    )[:, 0]
    plt.style.use(["default", "science"])
    plt.figure(figsize=(8, 6))
    plt.bar(
        sorted_compounds,
        [len(processed_data[compound]) for compound in sorted_compounds],
        color="tab:red",
        label="non-filtered",
    )
    plt.bar(
        sorted_compounds,
        [len(filtered_data[compound]) for compound in sorted_compounds],
        color="tab:blue",
        label="filtered",
    )
    min_count = (min([len(filtered_data[compound]) for compound in sorted_compounds]),)
    plt.axhline(
        min_count,
        color="black",
        ls="--",
        label=f"Min. {min_count}",
        linewidth=1.5,
    )
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.title("Dataset sizes after filtration", fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig("dataset_sizes_after_filtration.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    compounds = ["Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Ti", "V"]

    # ----------------------------------------
    # All raw feff data
    # ----------------------------------------
    if os.path.exists("feff_raw_all.pkl"): # <---- caching
        feff_raw_all = load(open("feff_raw_all.pkl", "rb"))
    else:
        feff_raw_all = p_map(lambda c: RAWDataFEFF(c), compounds)
        dump(feff_raw_all, open("feff_raw_all.pkl", "wb"))

    # ----------------------------------------
    # Process without emperical truncation
    # ----------------------------------------
    feff_all = p_map(
        lambda data: np.array(
            [
                FEFFData(
                    data.compound, data.parameters[id], id, do_transform=False
                ).transform(include_emperical_truncation=False)
                for id in data.ids
            ]
        ),
        feff_raw_all,
    )
    processed_data = {data[0].compound: data for data in feff_all}

    # ----------------------------------------
    # Energy range selection
    # ----------------------------------------
    (
        e_start,
        e_end,
    ) = find_e_start_and_end(compounds, processed_data)
    with open("e_starts.yaml", "w") as f:
        yaml.dump({k: str(v) for k, v in e_start.items()}, f, default_flow_style=False)

    # ----------------------------------------
    # Filter data where peak is outside the range
    # ----------------------------------------
    filtered_data = filter_off_peaks_data(compounds, processed_data, e_start, e_end)
    plot_count_of_filtered_data(compounds, processed_data, filtered_data)
