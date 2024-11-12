# %%
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy.stats as stats
from scipy import stats

from omnixas.scripts.plots.scripts import (
    AllDataTags,
    AllExpertMetrics,
    AllExpertModels,
    AllTunedMetrics,
    CompareAllExpertAndTuned,
    XASModelsOfCategory,
)
from src.data.ml_data import DataQuery

# %%

# NOTE: THE MSES DO NOT SATISFY THE NORMALITY ASSUMPTIONS OF THE PAIRED T-TEST


def compare_mses(
    mse_model1,
    mse_model2,
    alpha=0.05,
    test_method: Union[stats.wilcoxon, stats.ttest_rel] = stats.wilcoxon,
):

    assert len(mse_model1) == len(mse_model2), "The two lists must have the same length"

    # Convert lists to numpy arrays
    mse_model1 = np.array(mse_model1)
    mse_model2 = np.array(mse_model2)

    t_statistic, p_value = test_method(mse_model1, mse_model2, alternative="greater")

    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Interpret results
    if p_value < alpha:
        if t_statistic > 0:
            print("Model 2 is significantly better than Model 1")
        else:
            print("Model 1 is significantly better than Model 2")
    else:
        print("There is no statistically significant difference between the two models")

    return t_statistic, p_value


comparision = CompareAllExpertAndTuned()

# %%


t_stats = []
p_values = {}
for k, comp in comparision.items():
    compound = k.element
    sim = k.type

    mse_model1 = comp.metric1.mse_per_spectra
    mse_model2 = comp.metric2.mse_per_spectra

    print("-" * 40)
    print(f"Comparing models for {compound}")
    t_statistic, p_value = compare_mses(mse_model1, mse_model2)
    t_stats.append(t_statistic)
    key = compound if sim == "FEFF" else compound + "_" + sim
    p_values[key] = p_value

# plot t_stats and p_values
FONTSIZE = 18
fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
ax.bar(p_values.keys(), p_values.values(), color=plt.get_cmap("tab10").colors)

# # add values
# for i, v in enumerate(p_values.values()):
#     ax.text(i, v + 0.01, f"{v:.2f}", color="black", ha="center")

ax.set_yscale("log")
ax.axhline(0.05, color="gray", linestyle="--", label=r"$\alpha = 0.05$")
# put pair t_test label in y axis
ax.set_ylabel(r"Wilcoxon $p$-value", fontsize=FONTSIZE)
ax.set_xlabel("Element", fontsize=FONTSIZE)
x_tick_labels = [label.replace("_", "\n") for label in p_values.keys()]
ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=FONTSIZE * 0.7)
ax.tick_params(axis="y", labelsize=FONTSIZE * 0.7)
ax.legend(fontsize=FONTSIZE * 0.8)
plt.tight_layout()
plt.savefig("wilcoxon_test.pdf", bbox_inches="tight", dpi=300)


# %%
