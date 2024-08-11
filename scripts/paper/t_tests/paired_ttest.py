# %%
import scienceplots
from scipy.stats import shapiro, normaltest, anderson, kstest, chisquare
import scienceplots
from scipy.stats import ttest_rel, shapiro
from src.models.trained_models import MeanModel

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from config.defaults import cfg
from src.data.ml_data import load_xas_ml_data, DataQuery
from src.models.trained_models import Trained_FCModel
import matplotlib.pyplot as plt

# %%


# normality test needs to satistified, if not use wilcoxon test
# sharpio test is for normality test
# null hypothesis is that difference is nonrl
# p_value < 0.05, reject null hypothesis


for simulation_type in ["FEFF", "VASP"]:
    for compound in cfg.compounds:
        if simulation_type == "VASP" and compound not in ["Ti", "Cu"]:
            continue

        expert_model = Trained_FCModel(
            DataQuery(compound=compound, simulation_type=simulation_type),
            name="per_compound_tl",
        )
        tuned_universal_model = Trained_FCModel(
            DataQuery(compound=compound, simulation_type=simulation_type), name="ft_tl"
        )
        expert_mses = expert_model.mse_per_spectra
        tuned_universal_mses = tuned_universal_model.mse_per_spectra

        expert_mses = np.log(expert_mses)
        tuned_universal_mses = np.log(tuned_universal_mses)

        # boxcox transformation
        from scipy.stats import boxcox

        differences = expert_mses - tuned_universal_mses

        # plt.hist(differences, bins=50, density=True, histtype="step")
        plt.hist(
            expert_mses,
            bins=50,
            density=True,
            histtype="step",
            label=f"{compound}_{simulation_type}_expert",
        )
        plt.legend()

        # plt.show()

        # wilcoxon test
        print(
            f"compound: {compound}", "is normal", shapiro(differences).pvalue
        )  # all are NOT normal !! # CANNOT USE PAIRED T-TEST

# %%


# Geometric Mean: The geometric mean is a better measure of central tendency for
# log-normal distributions compared to the arithmetic mean. The geometric mean is
# calculated as the nth root of the product of the n values, where n is the
# number of observations.

# Ratio-based Comparisons: Since the distribution of MSE is log-normal, it is
# more appropriate to compare the ratios of MSE values across different models
# rather than the differences. This helps to account for the multiplicative
# nature of the log-normal distribution.


# %%

plt.style.use(["default", "science"])
fig, ax = plt.subplots(figsize=(8, 6))
stat_list = []
p_list = []
FONTSIZE = 18
colors = plt.cm.tab10.colors
for simulation_type in ["FEFF", "VASP"]:
    for i, compound in enumerate(cfg.compounds):
        if simulation_type == "VASP" and compound not in ["Ti", "Cu"]:
            continue
        expert_model = Trained_FCModel(
            DataQuery(compound=compound, simulation_type=simulation_type),
            name="per_compound_tl",
        )
        tuned_universal_model = Trained_FCModel(
            DataQuery(compound=compound, simulation_type=simulation_type), name="ft_tl"
        )
        expert_mses = expert_model.mse_per_spectra
        tuned_universal_mses = tuned_universal_model.mse_per_spectra

        # expert_mses = np.log(expert_mses)
        # tuned_universal_mses = np.log(tuned_universal_mses)

        # # wilcoxon test
        # statistics, p_val = stats.wilcoxon(
        #     expert_mses,
        #     tuned_universal_mses,
        #     alternative="greater",
        # )
        # statistics = statistics / len(expert_mses)

        # # scatter plot of mses
        # plt.scatter(expert_mses, tuned_universal_mses)
        # plt.show()
        # continue

        # differences = expert_mses - tuned_universal_mses

        # # paried t-test
        # statistics, p_val = ttest_rel(
        #     expert_mses,
        #     tuned_universal_mses,
        #     alternative="greater",
        # )

        # wilcoxon test
        statistics, p_val = stats.wilcoxon(
            expert_mses,
            tuned_universal_mses,
            alternative="greater",
        )

        # normalize statistics
        statistics = statistics / len(expert_mses)

        p_list.append(p_val)
        stat_list.append(statistics)

        print(f"compound: {compound}", statistics, p_val)

        plt.scatter(
            statistics,
            -np.log10(p_val),
            marker="s",
            s=500,
            color=colors[i],
            alpha=0.5,
        )

        # add annotation in middle of the square
        ax.text(
            statistics,
            -np.log10(p_val),
            compound if simulation_type == "FEFF" else f"{compound}_V",
            fontsize=FONTSIZE,
            ha="center",
            va="center",
            color="black",
        )

        ax.set_xlabel("Statistics", fontsize=FONTSIZE)
        ax.set_ylabel(r"$-\log_{10}(p_{val})$", fontsize=FONTSIZE)

        # # wilcoxon test
        # print(
        #     f"compound: {compound}",
        #     stats.wilcoxon(expert_mses, tuned_universal_mses, alternative="greater").pvalue
        #     < 0.05,
        #     "normlaity",
        #     shapiro(differences).pvalue < 0.05,
        #     normaltest(differences).pvalue < 0.05,
        #     anderson(differences, dist="norm").critical_values[2]
        #     < anderson(differences, dist="norm").statistic,
        #     kstest(differences, "norm").pvalue < 0.05,
        #     p_val < 0.05,
        #     p_val_wilcoxon < 0.05,
        # )

    plt.hlines(-np.log10(0.05), 0, max(stat_list), color="red")

# %%

fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
ax.bar(
    np.arange(len(p_list)),
    p_list,
    color=colors,
)
ax.set_yscale("log")
ax.set_xticks(np.arange(len(p_list)))
ax.set_xticklabels(cfg.compounds, fontsize=FONTSIZE)

# center
ax.set_ylabel(
    r"$p_{val}$ \\ Wilcoxon signed-rank (greater)",
    fontsize=FONTSIZE,
    labelpad=10,
    # rotation=0,
    ha="center",
)
ax.set_xlabel("Compound", fontsize=FONTSIZE)
ax.hlines(
    0.05,
    0 - 0.8,
    len(p_list),
    color="red",
    linestyle="--",
    label=r"$p_{val} = 0.05$",
)
ax.legend(fontsize=FONTSIZE * 0.8)

# %%

import numpy as np


def bootstrap_geometric_mean_test(
    mse_set_1, mse_set_2, num_bootstrap=10000, alpha=0.05
):
    """
    Perform a bootstrap hypothesis test to compare the geometric means of two sets of MSEs.

    Parameters:
    mse_set_1 (array-like): First set of MSEs.
    mse_set_2 (array-like): Second set of MSEs.
    num_bootstrap (int): Number of bootstrap samples to draw (default is 10,000).
    alpha (float): Significance level for the test (default is 0.05).

    Returns:
    p_value (float): The p-value for the hypothesis test.
    observed_diff (float): The observed difference in geometric means.
    """

    # Calculate the observed geometric means and their difference
    geo_mean_1 = np.exp(np.mean(np.log(mse_set_1)))
    geo_mean_2 = np.exp(np.mean(np.log(mse_set_2)))
    observed_diff = geo_mean_1 - geo_mean_2

    # Combine both sets for bootstrapping
    combined_set = np.concatenate([mse_set_1, mse_set_2])
    n1 = len(mse_set_1)

    # Bootstrap sampling
    bootstrap_diffs = []
    for _ in range(num_bootstrap):
        # Resample with replacement
        sample_1 = np.random.choice(combined_set, size=n1, replace=True)
        sample_2 = np.random.choice(
            combined_set, size=len(combined_set) - n1, replace=True
        )

        # Calculate geometric means for resampled sets
        geo_mean_1_boot = np.exp(np.mean(np.log(sample_1)))
        geo_mean_2_boot = np.exp(np.mean(np.log(sample_2)))

        # Store the difference in geometric means
        bootstrap_diffs.append(geo_mean_1_boot - geo_mean_2_boot)

    # Calculate p-value
    bootstrap_diffs = np.array(bootstrap_diffs)
    p_value = np.mean(bootstrap_diffs >= observed_diff)

    # Adjust p-value for two-tailed test if needed
    if observed_diff < 0:
        p_value = np.mean(bootstrap_diffs <= observed_diff)

    return p_value, observed_diff


bootstrap_tests = {}
for compound in cfg.compounds:
    for simulation_type in ["FEFF", "VASP"]:
        if simulation_type == "VASP" and compound not in ["Ti", "Cu"]:
            continue
        expert_model = Trained_FCModel(
            DataQuery(compound=compound, simulation_type=simulation_type),
            name="per_compound_tl",
        )
        tuned_universal_model = Trained_FCModel(
            DataQuery(compound=compound, simulation_type=simulation_type), name="ft_tl"
        )
        expert_mses = expert_model.mse_per_spectra
        tuned_universal_mses = tuned_universal_model.mse_per_spectra

        p_value, observed_diff = bootstrap_geometric_mean_test(
            expert_mses, tuned_universal_mses
        )
        print(f"compound: {compound}", p_value, observed_diff)
        bootstrap_tests[f"{compound}_{simulation_type}"] = (p_value, observed_diff)

# %%

bootstrap_tests
# plot pvalues in bar
fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
p_values = [bootstrap_tests[key][0] for key in bootstrap_tests.keys()]
ax.bar(
    np.arange(len(p_values)),
    p_values,
    color=colors,
)
ax.set_yscale("log")
ax.set_xticks(np.arange(len(p_values)))
ax.set_xticklabels(bootstrap_tests.keys(), fontsize=FONTSIZE, rotation=90)
ax.hlines(
    0.05,
    0 - 0.8,
    len(p_values),
    color="red",
    linestyle="--",
    label=r"$p_{val} = 0.05$",
)

# %%

# scatter plot of ovserberd diff and pvalues
fig, ax = plt.subplots(figsize=(8, 6))
plt.style.use(["default", "science"])
p_values = [bootstrap_tests[key][0] for key in bootstrap_tests.keys()]
observed_diffs = [bootstrap_tests[key][1] for key in bootstrap_tests.keys()]
ax.scatter(
    observed_diffs,
    -np.log10(p_values),
    color=colors,
)
ax.set_xlabel("Observed Difference in Geometric Means", fontsize=FONTSIZE)
ax.set_ylabel(r"$-\log_{10}(p_{val})$", fontsize=FONTSIZE)
for i, key in enumerate(bootstrap_tests.keys()):
    ax.text(
        observed_diffs[i],
        -np.log10(p_values[i]),
        key[:-5] if "FEFF" in key else f"{key[:-4]}_V",
        fontsize=FONTSIZE * 0.8,
        ha="center",
        va="center",
        color="black",
    )
