# %%

import scienceplots
from scipy.stats import ttest_rel, shapiro, ttest_rel_bootstrap

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from config.defaults import cfg
from src.data.ml_data import load_xas_ml_data, DataQuery
from src.models.trained_models import Trained_FCModel
import matplotlib.pyplot as plt

# %%

expertModel = Trained_FCModel(DataQuery("Cu", "FEFF"), name="per_compound_tl")
tunedUniversalModel = Trained_FCModel(DataQuery("Cu", "FEFF"), name="ft_tl")
# %%

plt.hist(expertModel.mse_per_spectra, alpha=0.4, label="expert")
plt.hist(tunedUniversalModel.mse_per_spectra, alpha=0.4, label="tuned universal")
plt.legend()

# %%

# mse_data = expertModel.mse_per_spectra
mse_data = expertModel.mse_per_spectra

# Assuming your MSE data is in a numpy array called 'mse_data'

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for j, mse_data in enumerate(
    [expertModel.mse_per_spectra, tunedUniversalModel.mse_per_spectra]
):
    # Test different distributions
    distributions = [
        ("Exponential", stats.expon),
        ("Gamma", stats.gamma),
        ("Log-normal", stats.lognorm),
        ("Weibull", stats.weibull_min),
    ]

    axes = axes.ravel()

    for i, (name, distribution) in enumerate(distributions):
        # Fit the distribution to the data
        params = distribution.fit(mse_data)

        # Calculate the PDF
        x = np.linspace(min(mse_data), max(mse_data), 100)
        pdf = distribution.pdf(x, *params)

        # Plot the histogram and the fitted PDF
        axes[i].hist(
            mse_data, bins=30, density=True, alpha=0.1, color="k" if j == 0 else "b"
        )
        axes[i].plot(
            x,
            pdf,
            "r-",
            lw=0.7,
            label=name,
            # alpha=0.6,
            color="r" if j == 0 else "b",
        )
        axes[i].set_title(name)
        axes[i].set_xlabel("MSE")
        axes[i].set_ylabel("Density")

plt.tight_layout()
plt.show()

# %%

import numpy as np
import pandas as pd


def analyze_qq_plots(qq_list):
    results = []
    distributions = ["Exponential", "Gamma", "Log-normal", "Weibull"]
    models = ["Model 1", "Model 2"]  # Generic names

    for i, dist in enumerate(distributions):
        for j, model in enumerate(models):
            qq = qq_list[i + j * 4]

            r_squared = qq[1][2] ** 2
            theoretical_q = qq[0][0]
            sample_q = qq[0][1]
            fitted_q = qq[1][0] * theoretical_q + qq[1][1]
            rmse = np.sqrt(np.mean((sample_q - fitted_q) ** 2))
            slope, intercept = qq[1][0], qq[1][1]

            results.append(
                {
                    "Distribution": dist,
                    "Model": model,
                    "R_squared": r_squared,
                    "RMSE": rmse,
                    "Slope": slope,
                    "Intercept": intercept,
                }
            )

    return pd.DataFrame(results)


# Analyze the QQ plots
results_df = analyze_qq_plots(qq_list)

# Display the results
print(results_df)

# Find the best fit for each model
best_fits = results_df.loc[results_df.groupby("Model")["R_squared"].idxmax()]
print("\nBest fits for each model:")
print(best_fits[["Model", "Distribution", "R_squared"]])

# Compare models directly
print("\nDirect model comparison:")
for dist in results_df["Distribution"].unique():
    model_r2_values = results_df[results_df["Distribution"] == dist]["R_squared"].values
    if len(model_r2_values) == 2:
        diff = model_r2_values[1] - model_r2_values[0]
        print(f"{dist}: Model 2 - Model 1 = {diff:.4f}")
    else:
        print(f"Insufficient data for {dist}")
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_qq_comparison(mse_expert, mse_tuned, distributions):
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    axs = axs.ravel()

    for ax, (name, dist) in zip(axs, distributions):
        # Fit distributions
        params_expert = dist.fit(mse_expert)
        params_tuned = dist.fit(mse_tuned)

        # Create Q-Q plots
        qq_expert = stats.probplot(
            mse_expert, dist=dist, sparams=params_expert[:-2], fit=True
        )
        qq_tuned = stats.probplot(
            mse_tuned, dist=dist, sparams=params_tuned[:-2], fit=True
        )

        # Plot expert model
        ax.plot(
            qq_expert[0][0],
            qq_expert[0][1],
            "bo",
            markersize=3,
            alpha=0.5,
            label="Expert Model",
        )
        ax.plot(
            qq_expert[0][0],
            qq_expert[1][0] * qq_expert[0][0] + qq_expert[1][1],
            "b-",
            lw=1,
        )

        # Plot tuned universal model
        ax.plot(
            qq_tuned[0][0],
            qq_tuned[0][1],
            "ro",
            markersize=3,
            alpha=0.5,
            label="Tuned Universal Model",
        )
        ax.plot(
            qq_tuned[0][0], qq_tuned[1][0] * qq_tuned[0][0] + qq_tuned[1][1], "r-", lw=1
        )

        # Set labels and title
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"{name} Q-Q Plot")
        ax.legend()

    plt.tight_layout()
    plt.show()


mse_expert = expertModel.mse_per_spectra
mse_tuned = tunedUniversalModel.mse_per_spectra

mse_expert = np.log(mse_expert)
mse_tuned = np.log(mse_tuned)

# Assuming you have mse_expert and mse_tuned as numpy arrays
# mse_expert = expertModel.mse_per_spectra
# mse_tuned = tunedUniversalModel.mse_per_spectra

# List of distributions to test
distributions = [
    ("Exponential", stats.expon),
    ("Gamma", stats.gamma),
    ("Log-normal", stats.lognorm),
    ("Weibull", stats.weibull_min),
]

# Create the plots
plot_qq_comparison(mse_expert, mse_tuned, distributions)
# %%

compounds = cfg.compounds

# plot 2,4 subplots for each compound for expert and tuned universal model for log-normal onlyt
fig, axs = plt.subplots(4, 2, figsize=(8, 16))
axs = axs.ravel()
for i, compound in enumerate(compounds):
    expertModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="per_compound_tl")
    tunedUniversalModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="ft_tl")
    mse_expert = expertModel.mse_per_spectra
    mse_tuned = tunedUniversalModel.mse_per_spectra
    mse_expert = np.log(mse_expert)
    mse_tuned = np.log(mse_tuned)
    dist_name = "Log-normal"
    dist = stats.lognorm
    params_expert = dist.fit(mse_expert)
    params_tuned = dist.fit(mse_tuned)
    qq_expert = stats.probplot(
        mse_expert, dist=dist, sparams=params_expert[:-2], fit=True
    )
    qq_tuned = stats.probplot(mse_tuned, dist=dist, sparams=params_tuned[:-2], fit=True)
    ax = axs[i]
    ax.plot(
        qq_expert[0][0],
        qq_expert[0][1],
        "bo",
        markersize=3,
        alpha=0.5,
        label="Expert Model",
    )
    ax.plot(
        qq_expert[0][0],
        qq_expert[1][0] * qq_expert[0][0] + qq_expert[1][1],
        "b-",
        lw=1,
    )
    ax.plot(
        qq_tuned[0][0],
        qq_tuned[0][1],
        "ro",
        markersize=3,
        alpha=0.5,
        label="Tuned Universal Model",
    )
    ax.plot(
        qq_tuned[0][0], qq_tuned[1][0] * qq_tuned[0][0] + qq_tuned[1][1], "r-", lw=1
    )
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"{compound} {dist_name} Q-Q Plot")
    ax.legend()
plt.tight_layout()
plt.show()

# %%

compound = "Cu"

fig, axs = plt.subplots(4, 2, figsize=(8, 16), sharey=True, sharex=True)
for ax, compound in zip(axs.ravel(), compounds):
    expertModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="per_compound_tl")
    tunedUniversalModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="ft_tl")

    univ_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
    data = load_xas_ml_data(DataQuery(compound, "FEFF"))
    univ_model.data = data

    BINS = 100

    mse_expert = expertModel.mse_per_spectra
    mse_tuned = tunedUniversalModel.mse_per_spectra
    mse_universal = univ_model.mse_per_spectra

    mse_expert = np.log(mse_expert)
    mse_tuned = np.log(mse_tuned)
    mse_universal = np.log(mse_universal)

    # # edges only and no fill. step hist
    # ax.hist(mse_expert, bins=BINS, label="expert", histtype="step", density=True)
    # ax.hist(
    #     mse_tuned, bins=BINS, label="tuned universal", histtype="step", density=True
    # )
    # ax.hist(mse_universal, bins=BINS, label="universal", histtype="step", density=True)

    X = np.linspace(
        min([min(mse_expert), min(mse_tuned), min(mse_universal)]),
        max([max(mse_expert), max(mse_tuned), max(mse_universal)]),
        100,
    )

    mean = np.mean(mse_expert)
    std = np.std(mse_expert)
    # x_expert = np.linspace(min(mse_expert), max(mse_expert), 100)
    x_expert = X
    y_expert = stats.norm.pdf(x, mean, std)
    ax.plot(x_expert, y_expert, label="expert normal", color="blue", linestyle="--")

    mean = np.mean(mse_tuned)
    std = np.std(mse_tuned)
    # x_tuned = np.linspace(min(mse_tuned), max(mse_tuned), 100)
    x_tuned = X
    y_tuned = stats.norm.pdf(x, mean, std)
    ax.plot(x_tuned, y_tuned, label="tuned universal normal", color="green")

    mean = np.mean(mse_universal)
    std = np.std(mse_universal)
    x_universal = X
    y_universal = stats.norm.pdf(x, mean, std)
    ax.plot(x_universal, y_universal, label="universal normal", color="red")

    # # fill between two normal distributions
    # ax.fill_between(x_expert, y_expert, y_tuned, color="yellow", alpha=0.5)

    # ax.set_xlim(0, 0.025)

    # # cumulative hist
    # ax.hist(
    #     mse_expert,
    #     bins=BINS,
    #     label="expert",
    #     cumulative=True,
    #     alpha=0.5,
    #     histtype="step",
    # )
    # ax.hist(
    #     mse_tuned,
    #     bins=BINS,
    #     label="tuned universal",
    #     cumulative=True,
    #     alpha=0.5,
    #     histtype="step",
    # )

    ax.set_title(compound)
    # ax.legend()
axs[-1, -1].legend()

# %%

# find confidence interval for mse
import scipy.stats as stats

for compound in cfg.compounds:
    expertModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="per_compound_tl")
    tunedUniversalModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="ft_tl")

    univ_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
    data = load_xas_ml_data(DataQuery(compound, "FEFF"))
    univ_model.data = data

    mse_expert = expertModel.mse_per_spectra
    mse_tuned = tunedUniversalModel.mse_per_spectra
    mse_universal = univ_model.mse_per_spectra

    mse_expert = np.log(mse_expert)
    mse_tuned = np.log(mse_tuned)
    mse_universal = np.log(mse_universal)

    # find confidence interval for mse
    mean_expert = np.mean(mse_expert)
    std_expert = np.std(mse_expert)
    n_expert = len(mse_expert)
    z = 1.95  # 95% confidence interval
    ci_expert = z * (std_expert / np.sqrt(n_expert))

    mean_tuned = np.mean(mse_tuned)
    std_tuned = np.std(mse_tuned)
    n_tuned = len(mse_tuned)
    ci_tuned = z * (std_tuned / np.sqrt(n_tuned))

    mean_universal = np.mean(mse_universal)
    std_universal = np.std(mse_universal)
    n_universal = len(mse_universal)
    ci_universal = z * (std_universal / np.sqrt(n_universal))

    # forest plot
    fig, ax = plt.subplots()
    ax.errorbar(1, mean_expert, yerr=ci_expert, fmt="o", label="expert")
    ax.errorbar(2, mean_tuned, yerr=ci_tuned, fmt="o", label="tuned universal")
    ax.errorbar(3, mean_universal, yerr=ci_universal, fmt="o", label="universal")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["expert", "tuned universal", "universal"])
    ax.set_ylabel("MSE")
    ax.set_title(compound)
    ax.legend()

# %%

# use bootstrap to find confidence interval for mse

from sklearn.utils import resample

for compound in cfg.compounds:
    expertModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="per_compound_tl")
    tunedUniversalModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="ft_tl")

    univ_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
    data = load_xas_ml_data(DataQuery(compound, "FEFF"))
    univ_model.data = data

    mse_expert = expertModel.mse_per_spectra
    mse_tuned = tunedUniversalModel.mse_per_spectra
    mse_universal = univ_model.mse_per_spectra

    mse_expert = np.log(mse_expert)
    mse_tuned = np.log(mse_tuned)
    mse_universal = np.log(mse_universal)

    # find confidence interval for mse
    n_expert = len(mse_expert) // 10
    n_tuned = len(mse_tuned) // 10
    n_universal = len(mse_universal) // 10

    n_iter = 1000
    ci_expert = np.zeros((n_iter, 2))
    ci_tuned = np.zeros((n_iter, 2))
    ci_universal = np.zeros((n_iter, 2))

    for i in range(n_iter):
        mse_expert_resampled = resample(mse_expert, n_samples=n_expert)
        mse_tuned_resampled = resample(mse_tuned, n_samples=n_tuned)
        mse_universal_resampled = resample(mse_universal, n_samples=n_universal)

        mean_expert = np.mean(mse_expert_resampled)
        ci_expert[i] = [
            mean_expert - 1.96 * np.std(mse_expert_resampled),
            mean_expert + 1.96 * np.std(mse_expert_resampled),
        ]

        mean_tuned = np.mean(mse_tuned_resampled)
        ci_tuned[i] = [
            mean_tuned - 1.96 * np.std(mse_tuned_resampled),
            mean_tuned + 1.96 * np.std(mse_tuned_resampled),
        ]

        mean_universal = np.mean(mse_universal_resampled)
        ci_universal[i] = [
            mean_universal - 1.96 * np.std(mse_universal_resampled),
            mean_universal + 1.96 * np.std(mse_universal_resampled),
        ]
    ax.errorbar(1, np.mean(ci_expert), yerr=0, fmt="o", label="expert")
    ax.errorbar(2, np.mean(ci_tuned), yerr=0, fmt="o", label="tuned universal")
    ax.errorbar(3, np.mean(ci_universal), yerr=0, fmt="o", label="universal")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["expert", "tuned universal", "universal"])
    ax.set_ylabel("MSE")
    ax.set_title(compound)
    ax.legend()

    break

# %%
import numpy as np


def fast_bootstrap_stats(data, num_iterations, statistic):
    n = len(data)
    # Generate all random indices at once
    indices = np.random.randint(0, n, size=(num_iterations, n))
    # Use these indices to create all resamples at once
    resamples = data[indices]
    # Apply the statistic to each resample
    return np.apply_along_axis(statistic, 1, resamples)


def bootstrap_ci(data, num_iterations=100000, statistic=np.mean, confidence_level=0.95):

    bootstrapped_stats = fast_bootstrap_stats(data, num_iterations, statistic)

    # Sort the bootstrapped statistics
    bootstrapped_stats.sort()

    # Find the confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2
    upper_percentile = 1 - (alpha / 2)

    lower_bound = np.percentile(bootstrapped_stats, lower_percentile * 100)
    upper_bound = np.percentile(bootstrapped_stats, upper_percentile * 100)

    return lower_bound, upper_bound


mses = {}
for compound in cfg.compounds:
    expertModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="per_compound_tl")
    tunedUniversalModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="ft_tl")

    univ_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
    data = load_xas_ml_data(DataQuery(compound, "FEFF"))
    univ_model.data = data

    mse_expert = expertModel.mse_per_spectra
    mse_tuned = tunedUniversalModel.mse_per_spectra
    mse_universal = univ_model.mse_per_spectra

    # mse_expert = np.log(mse_expert)
    # mse_tuned = np.log(mse_tuned)
    # mse_universal = np.log(mse_universal)

    mses[compound] = (mse_expert, mse_tuned, mse_universal)
# %%

# %%
import numpy as np
from scipy import stats


def compare_mses(mse_model1, mse_model2, alpha=0.05):
    # Ensure the lists have the same length
    assert len(mse_model1) == len(mse_model2), "The two lists must have the same length"

    # Convert lists to numpy arrays
    mse_model1 = np.array(mse_model1)
    mse_model2 = np.array(mse_model2)

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(mse_model1, mse_model2)

    # Calculate mean difference
    mean_diff = np.mean(mse_model1 - mse_model2)

    # Print results
    # print(f"Mean MSE for Model 1: {np.mean(mse_model1):.4f}")
    # print(f"Mean MSE for Model 2: {np.mean(mse_model2):.4f}")
    # print(f"Mean difference (Model 1 - Model 2): {mean_diff:.4f}")
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


universal_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
t_stats = []
p_values = []
for compound in cfg.compounds:
    expertModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="per_compound_tl")
    tunedUniversalModel = Trained_FCModel(DataQuery(compound, "FEFF"), name="ft_tl")

    mse_model1 = expertModel.mse_per_spectra
    mse_model2 = tunedUniversalModel.mse_per_spectra

    # universal_model.data = load_xas_ml_data(DataQuery(compound, "FEFF"))
    # mse_model2 = universal_model.mse_per_spectra

    print("-" * 40)
    print(f"Comparing models for {compound}")
    t_statistic, p_value = compare_mses(mse_model1, mse_model2)
    t_stats.append(t_statistic)
    p_values.append(p_value)
    print("-" * 40)

# %%

# plot t_stats and p_values
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].bar(cfg.compounds, t_stats)
ax[0].set_title("t-statistics")
ax[1].bar(cfg.compounds, p_values)
ax[1].set_title("p-values")
ax[1].axhline(0.05, color="red", linestyle="--")
ax[1].set_yscale("log")
plt.tight_layout()
plt.show()

# %%


def plot_t_test_results(elements, t_stats, p_values):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Sort data by absolute t-statistic
    # sorted_indices = np.argsort(np.abs(t_stats))[::-1]
    sorted_indices = np.arange(len(elements))
    elements = np.array(elements)[sorted_indices]
    t_stats = np.array(t_stats)[sorted_indices]
    p_values = np.array(p_values)[sorted_indices]

    y_pos = np.arange(len(elements))

    # Plot t-statistics
    ax1.barh(y_pos, t_stats, align="center")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(elements)
    ax1.axvline(x=0, color="gray", linestyle="--")
    ax1.set_xlabel("t-statistic")
    ax1.set_title("T-statistics")

    # Plot p-values
    ax2.barh(y_pos, -np.log10(p_values), align="center")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Remove y-axis labels for the second plot
    ax2.axvline(x=-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
    ax2.set_xlabel("-log10(p-value)")
    ax2.set_title("P-values")

    # Add significance threshold line and text
    ax2.text(
        -np.log10(0.05), len(elements), "p=0.05", color="red", ha="center", va="bottom"
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)  # Reduce space between subplots

    # Add an overall title
    fig.suptitle("T-statistics and P-values for Elements", fontsize=16, y=1.05)

    plt.show()


plot_t_test_results(elements, t_stats, p_values)
# %%

import matplotlib.pyplot as plt
import numpy as np


def create_volcano_plot(elements, t_stats, p_values):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert p-values to -log10(p)
    log_p = -np.log10(p_values)

    # Plot points
    scatter = ax.scatter(t_stats, log_p, s=60, alpha=0.7)

    # Add element labels to points
    for i, txt in enumerate(elements):
        ax.annotate(
            txt,
            (t_stats[i], log_p[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # Add threshold lines
    ax.axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=-2, color="blue", linestyle="--", alpha=0.5)
    ax.axvline(x=2, color="blue", linestyle="--", alpha=0.5)

    # Customize the plot
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Volcano Plot of T-test Results")

    # Add text for quadrants
    ax.text(
        0.98,
        0.98,
        "High t-stat\nLow p-value",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.text(
        0.02,
        0.98,
        "Low t-stat\nLow p-value",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.text(
        0.98,
        0.02,
        "High t-stat\nHigh p-value",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.text(
        0.02,
        0.02,
        "Low t-stat\nHigh p-value",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


# Example usage
elements = cfg.compounds

create_volcano_plot(elements, t_stats, p_values)

# %%

mses["Cu"][0]
p_dict = {}
for compound in cfg.compounds:
    print(compound)
    array1 = mses[compound][0]
    array2 = mses[compound][1]

    # array1 = np.log(array1)
    # array2 = np.log(array2)

    # print(
    #     "Data normality test",
    #     shapiro(array1).pvalue < 0.05,
    #     shapiro(array2).pvalue < 0.05,
    # )

    differences = array1 - array2
    # t test for each spectra
    p_values = []
    for diff in differences:
        t_statistic, p_value = ttest_rel(diff, np.zeros_like(diff))
        p_values.append(p_value)
    p_dict[compound] = np.array(p_values)

    t_statistic, p_value = ttest_rel(array1, array2, alternative="greater")

    print(t_statistic, p_value, p_value < 0.05)
    print("-" * 40)


# mse_expert = expertModel.mse_per_spectra
# mse_tuned = tunedUniversalModel.mse_per_spectra
# ttest_rel(mse_expert, mse_tuned)

# %%
# Wilcoxon signed-rank test
from scipy.stats import wilcoxon

t_stats = []
p_values = []
for compound in cfg.compounds:
    print(compound)
    array1 = mses[compound][0]
    array2 = mses[compound][1]

    # t_statistic, p_value = wilcoxon(array1, array2, alternative="greater")
    t_statistic, p_value = ttest_rel(array1, array2, alternative="greater")
    print(t_statistic, p_value, p_value < 0.05)
    t_stats.append(t_statistic)
    p_values.append(p_value)
    print("-" * 40)

# %%

# volcano plot
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))

# Convert p-values to -log10(p)
log_p = -np.log10(p_values)

# Plot points
# scatter = ax.scatter(t_stats, log_p, s=60, alpha=0.7)

# # colors tab10
colors = plt.cm.tab10.colors
# ax.bar(cfg.compounds, p_values, color=colors)
# ax.set_yscale("log")
# ax.hlines(0.05, -10, 10, color="red", linestyle="--")


FONTSIZE = 18
ax.hlines(-np.log10(0.05), -10, 10, color="red", linestyle="--")
for compound in cfg.compounds:

    # # color by data size
    # data = load_xas_ml_data(DataQuery(compound, "FEFF")).train.X
    # s = data.shape[0] / 50

    ax.scatter(
        t_stats[cfg.compounds.index(compound)],
        log_p[cfg.compounds.index(compound)],
        s=s,
        alpha=0.7,
    )
    ax.annotate(
        compound,
        (t_stats[cfg.compounds.index(compound)], log_p[cfg.compounds.index(compound)]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("t-statistic", fontsize=FONTSIZE)
    ax.set_ylabel(r"$-log_{10}(p-value)$", fontsize=FONTSIZE)


# %%
