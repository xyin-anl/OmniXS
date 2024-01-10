import scienceplots
from p_tqdm import p_map
from src.data.feff_data_raw import RAWDataFEFF
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # ----------------------------------------
    # FEF data for all compounds
    # ----------------------------------------
    compounds = ["Co", "Cr", "Cu", "Fe", "Mn", "Ni", "Ti", "V"]
    data = p_map(lambda c: RAWDataFEFF(c), compounds)

    # -------------------------------
    # data count for all compounds
    # -------------------------------
    plt.style.use(["default", "science"])
    fig = plt.figure(figsize=(8, 6))
    counts = {c: len(d) for c, d in zip(compounds, data)}
    counts = {
        k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    }

    # -------------------------------
    # bar plot of counts
    # -------------------------------
    plt.bar(
        counts.keys(),
        counts.values(),
        color="tab:green",
        edgecolor="green",
        linewidth=2,
        label="FEFF data count",
    )
    # -------------------------------
    # text on top of bars showing
    # -------------------------------
    for i, (compound, count) in enumerate(counts.items()):
        plt.text(
            i,
            count + 0.01 * max(counts.values()),
            f"{compound}\n{count/1000:.1f}k",
            ha="center",
            fontsize=16,
            color="black",
            fontweight="bold",
        )
    # -------------------------------
    # horizontal line at minimum
    # -------------------------------
    plt.axhline(
        min(counts.values()),
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        alpha=0.5,
        label=f"Minimum Count: {min(counts.values())}",
    )
    # -------------------------------
    # plot elements
    # -------------------------------
    plt.xlabel("Compound", fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel("Count", fontsize=18)
    plt.yticks(plt.yticks()[0], [f"{i/1000:.0f}k" for i in plt.yticks()[0]])
    plt.ylim([0, 1.1 * max(counts.values())])
    plt.title("Available data for each compound", fontsize=20)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig("available_data_count.pdf", dpi=300, bbox_inches="tight")
    plt.show()
