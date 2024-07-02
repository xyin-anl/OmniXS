# %%
import warnings
import os
import pickle
from p_tqdm import p_map
from typing import Literal
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from copy import deepcopy
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from scipy.fft import fft2, fftshift
from config.defaults import cfg
import matplotlib.pyplot as plt
import numpy as np
from src.data.ml_data import load_xas_ml_data, DataQuery, DataSplit, MLSplits
from src.models.trained_models import (
    Trained_FCModel,
    ElastNet,
    XGBReg,
    MeanModel,
    LinReg,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %%

simulation_type = "SOAP"
ml_splits = {c: load_xas_ml_data(DataQuery(c, simulation_type)) for c in cfg.compounds}
# ml_splits = {"ALL": load_xas_ml_data(DataQuery("ALL", "FEFF"))}


# %%

# # LINEAR REGRESSION
# def get_ml_models(model_class, ml_splits):
#     train_splits = {c: ml_split.train for c, ml_split in ml_splits.items()}
#     test_splits = {c: ml_split.test for c, ml_split in ml_splits.items()}
#     models = {
#         c: model_class().fit(train_splits[c].X, train_splits[c].y)
#         for c in cfg.compounds
#     }
#     mses_model = {
#         c: mean_squared_error(test_splits[c].y, model.predict(test_splits[c].X))
#         for c, model in models.items()
#     }
#     mses_mean = {
#         c: mean_squared_error(
#             test_splits[c].y,
#             np.full_like(test_splits[c].y, np.mean(train_splits[c].y, axis=0)),
#         )
#         for c in cfg.compounds
#     }
#     mses_rel = {
#         c: mses_mean / mses_model
#         for c, mses_mean, mses_model in zip(
#             cfg.compounds, mses_mean.values(), mses_model.values()
#         )
#     }
#     return models, mses_rel
# ml_models, mses_rel = get_ml_models(LinearRegression, ml_splits)


# %%


# WRAPPER FOR LINEAR REGRESSION MODELS USING STATSMODELS FOR CONFIDENCE INTERVALS
class DummyLinearModel:
    def __init__(self, wls_models: np.ndarray):
        self._models = wls_models
        self._class_name = wls_models[0].__class__.__name__
        self._coefficients = np.array([x.params for x in wls_models]).squeeze()

    @property
    def coef_(self):
        return self._coefficients

    @property
    def models(self):
        return self._models

    @property
    def __class__(self):
        return type(self._class_name, (object,), {})

    def predict(self, X):
        return np.array([x.predict(X) for x in self.models]).T

    @property
    def confidence_intervals(self):
        return np.array([x.conf_int() for x in self.models])

    def mse(self, splits):
        y_pred = self.predict(splits.test.X)
        mse = mean_squared_error(splits.test.y, y_pred)
        return mse

    def mean_mse(self, splits):
        y_pred = np.full_like(splits.test.y, np.mean(splits.train.y, axis=0))
        mse = mean_squared_error(splits.test.y, y_pred)
        return mse

    def mse_rel_mean(self, splits):
        return self.mean_mse(splits) / self.mse(splits)

    def pruned_coef_(self, fill_value=0):
        coef = deepcopy(self.coef_)
        conf_intervals = self.confidence_intervals
        for t_idx in range(coef.shape[1]):
            for f_idx in range(coef.shape[0]):
                if (
                    conf_intervals[f_idx, t_idx, 0] * conf_intervals[f_idx, t_idx, 1]
                    > 0
                ):
                    coef[f_idx, t_idx] = fill_value
        return coef


def get_sm_linear_models(ml_splits):
    sm_models = {
        c: DummyLinearModel(
            np.array(
                [
                    sm.OLS(ml_split.train.y[:, i], ml_split.train.X).fit()
                    for i in range(ml_split.train.y.shape[1])
                ]
            )
        )
        for c, ml_split in ml_splits.items()
    }
    mses_rel = {
        c: model.mse_rel_mean(ml_split)
        for c, model, ml_split in zip(
            sm_models.keys(),
            sm_models.values(),
            ml_splits.values(),
        )
    }
    return sm_models, mses_rel


ml_models, mses_rel = get_sm_linear_models(ml_splits)
# ml_models, mses_rel = get_sm_linear_models(pca_reduced_ml_splits)

# %%


def get_weights_with_CI(linear_model, top_n=5, ax=None):
    weights = linear_model.coef_
    conf_intervals = linear_model.confidence_intervals
    avg_weights = np.mean(weights, axis=0)
    top_n_idx = np.argsort(np.abs(avg_weights))[-top_n:][::-1]
    top_weights = weights[top_n_idx]
    top_ci = conf_intervals[top_n_idx]
    w_ci = np.array(
        [
            (w, ci_l, ci_u)
            for w, (ci_l, ci_u) in zip(top_weights.flatten(), top_ci.reshape(-1, 2))
        ]
    )
    w_ci = w_ci.reshape(top_n, -1, 3)
    return w_ci, top_n_idx


# w_ci, _ = get_weights_with_CI(ml_models["ALL"], top_n=64)
# significat_features = {}
# for feature_idx, feature in enumerate(w_ci):
#     if np.any(feature[:, 1] * feature[:, 2] > 0):
#         # count and store the feature index
#         significat_features[feature_idx] = np.sum(feature[:, 1] * feature[:, 2] > 0)
# plt.bar(significat_features.keys(), significat_features.values())


# do above for all compounds
def plot_significant_weights(ml_models, mses_rel, ml_plits):
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    for ax, (c, model) in zip(axs.flatten(), ml_models.items()):
        w_ci, _ = get_weights_with_CI(
            # model, top_n=pca_reduced_ml_splits[c].train.X.shape[1]
            model,
            top_n=ml_plits[c].train.X.shape[1],
        )
        significat_features = {}
        for feature_idx, feature in enumerate(w_ci):
            if np.any(feature[:, 1] * feature[:, 2] > 0):
                significat_features[feature_idx] = np.sum(
                    feature[:, 1] * feature[:, 2] > 0
                )
        ax.bar(significat_features.keys(), significat_features.values())
        ax.set_title(f"{c} {model.__class__.__name__} {mses_rel[c]:.2f}")
        # set text that gives the sum of significant features
        total_significant_weights = np.sum(list(significat_features.values()))
        ax.text(
            0.5,
            0.8,
            f"Significant Weigths: {total_significant_weights}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=30,
            color="red",
        )
    plt.tight_layout()


plot_significant_weights(ml_models, mses_rel, ml_splits)


# %%


def plot_top_weights_CI(model, top_n=5, ax=None):
    w_ci, top_n_idx = get_weights_with_CI(model, top_n=top_n)

    if ax is None:
        fig, ax = plt.subplots()

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(top_n)]

    for feature in range(top_n):
        for i in range(w_ci.shape[1]):
            ci = w_ci[feature, i, 1:]
            mean = w_ci[feature, i, 0]
            color = colors[feature] if ci[0] * ci[1] > 0 else "gray"

            ax.errorbar(
                i,
                mean,
                yerr=[[mean - ci[0]], [ci[1] - mean]],
                fmt="o",
                color=color,
                label=f"Feature {top_n_idx[feature]}",
            )
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    unique_labels["CI includes 0"] = plt.Line2D([0], [0], color="gray", lw=2)
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Weight")


fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(4, 2, hspace=0, wspace=0)
axs = gs.subplots(sharex=True)
# put mse in title as well
for ax, (c, model) in zip(axs.flatten(), ml_models.items()):
    plot_top_weights_CI(model, ax=ax, top_n=5)
    ax.set_title(f"{c} {model.__class__.__name__} {mses_rel[c]:.2f}")


# %%


def plot_feature_weights(
    linear_models,
    ml_splits,
    mses_rel,
    axs=None,
    save_file_postfix="",
    prune_weights=False,
    plot_type: Literal["fft_2d", "fft_1d", "fft_1d_mean", "weights"] = "weights",
):

    ASPECT_RATIO = 16 / 9
    WIDTH = 20
    HEIGHT = WIDTH * ASPECT_RATIO
    if axs is None:
        fig = plt.figure(figsize=(WIDTH, HEIGHT))
        gs = fig.add_gridspec(4, 2)  # , hspace=-0.5, wspace=0.1)
        axs = gs.subplots()

    for ax, compound in zip(axs.flatten(), linear_models.keys()):

        model = linear_models[compound]

        title = simulation_type
        title += f" {compound} {model.__class__.__name__} {mses_rel[compound]:.2e}"

        if plot_type == "fft_1d":

            coeffs = deepcopy(model.coef_)
            coeffs_fft = np.abs(np.fft.fft(coeffs, axis=1))
            ax.plot(coeffs_fft)

        elif plot_type == "fft_1d_mean":
            coeffs = (
                deepcopy(model.coef_) if not prune_weights else model.pruned_coef_()
            )
            coeffs_fft = np.abs(np.fft.fft(coeffs, axis=1))
            ax.plot(np.mean(coeffs_fft, axis=0))

        elif plot_type == "fft_2d":
            coeffs = deepcopy(model.coef_)  # DOESNT PRUNE WEIGHTS
            coeffs_fft = fftshift(fft2(coeffs))
            img = np.log(np.abs(coeffs_fft))
            ax.imshow(
                img, cmap="jet", aspect=(img.shape[1] / img.shape[0]) / ASPECT_RATIO
            )

        elif plot_type == "weights":
            img = deepcopy(model.coef_)
            ax.imshow(
                img, cmap="jet", aspect=(img.shape[1] / img.shape[0]) / ASPECT_RATIO
            )

        elif plot_type == "pruned_weights":
            # Get the coloring from the unpruned weights
            unpruned_weights = deepcopy(model.coef_)
            pruned_weights = model.pruned_coef_(fill_value=np.nan)

            # Calculate color limits based on the unpruned weights
            vmin = np.nanmin(unpruned_weights)
            vmax = np.nanmax(unpruned_weights)

            # Plot the pruned weights with the same color scaling
            img = deepcopy(pruned_weights)
            im = ax.imshow(
                pruned_weights,
                cmap="jet",
                aspect=(img.shape[1] / img.shape[0]) / ASPECT_RATIO,
                vmin=vmin,  # Set the same minimum color limit
                vmax=vmax,  # Set the same maximum color limit
            )
            cbar = plt.colorbar(im, orientation="horizontal", ax=ax)
            cbar.set_label("Weights")

        else:
            raise ValueError("Invalid plot type")

        ax.set_title(title, fontsize="xx-large")
        ax.tick_params(axis="both", which="major", labelsize="xx-large")

        # plt.colorbar(ax.images[0], orientation="horizontal", ax=ax, shrink=0.4)

    plt.tight_layout()
    name = simulation_type
    name += f"_{plot_type}"
    name += f"_{model.__class__.__name__}"
    name += f"_{save_file_postfix}" if save_file_postfix else ""
    plt.savefig(f"{name}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{name}.png", bbox_inches="tight", dpi=300)
    return axs


def plot_all_feature_weights_types(**kwargs):
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(len(cfg.compounds), 4, hspace=0, wspace=0.1)

    for plot_type in ["weights", "fft_2d", "fft_1d", "fft_1d_mean"]:
        # for plot_type in ["weights", "pruned_weights"]:
        axs_out = plot_feature_weights(**kwargs, plot_type=plot_type)
        for ax in axs_out.flatten():
            plt.figure()
            plt.show(ax.figure)  # Show each axs_out figure individually

        for i, ax in enumerate(axs_out.flatten()):
            sub_fig = fig.add_subplot(gs[i // 4, i % 4])
            ax.figure = fig
            fig.add_axes(ax)
            ax.set_position(sub_fig.get_position(fig))

    plt.tight_layout()
    fig.savefig(
        f"{simulation_type}_feature_weights_all.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show(fig)  # Show the combined figure
    return axs_out


plot_all_feature_weights_types(
    linear_models=ml_models,
    ml_splits=ml_splits,
    mses_rel=mses_rel,
    save_file_postfix="",
)


# %%


def plot_top_n_features(top_n, models, mses, axs=None):
    if axs is None:
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(4, 2, hspace=0, wspace=0.1)
        axs = gs.subplots()
    compound_idxs = {}
    for ax, compound in zip(axs.flatten(), models.keys()):
        model = models[compound]
        mse = mses[compound]
        coef_per_fea = np.sum(np.abs(model.coef_), axis=0)

        selected_idx = coef_per_fea.argsort()[-top_n:][::-1]
        compound_idxs[compound] = selected_idx
        for idx in selected_idx:
            ax.plot(model.coef_.T[idx], label=f"idx={idx}")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize="medium")
        title = simulation_type
        title += f"{compound} {mse:.2f} {model.__class__.__name__} {mse:.2f}"
        ax.set_title(title, fontsize="xx-large")

    plt.tight_layout()
    return axs


plot_top_n_features(top_n=10, models=ml_models, mses=mses_rel)

# %%


# lets set the assumptions for linreg

# meaning

# Numerical feature: Increasing the numerical feature by one unit changes the
# estimated outcome by its weight.

# Confidence interval: Estimated weights come with confidence intervals. A
# confidence interval is a range for the weight estimate that covers the “true”
# weight with a certain confidence. For example, a 95% confidence interval for
# a weight of 2 could range from 1 to 3. The interpretation of this interval
# would be: If we repeated the estimation 100 times with newly sampled data,
# the confidence interval would include the true weight in 95 out of 100 cases,
# given that the linear regression model is the correct model for the data.


# ASSUMPTIONS IN LINEAR REGRESSION

# 1. Linear relationship: if on-linear add interaction terms or use regression splines

# 2. Normality: if violated, confidence intervals are invalid. How does this
# change for target with multiple dimension ?

# 3. Homoscedasticity: constant variance of the residuals. But in spectras, we
# know that the variance changes based on energy grids (based on magnitude of
# the spectras?).. this is variance in data, so does this mean teh variance in residuals?

# 4. Independendece: for each observation (what about feature dimensions)?
# maybe the instance for smae material id  but different sites are similar?

# 5. Fixed features: features are fixed and not random variables, i.e. no
# measurement error. In our case, features are deterministic but has uncertainty.

# 6. No multicollinearity: features are not correlated with each other. Else
# interpretation is difficult.

# DEFINITIONS

# t-staitstic: Weight estimate divided by its standard error giving less
# importance to features that we are not sure about.


# WEIGHT PLOT

# if weights are close to zero and confidence intervals include zero, then the
# feature importance is not statistically significant. Some can be close to
# zero but with narrow confidence intervals, then they are statistically
# significant.

# So you can remove points in weight matrix that are not statistically significant.

# EFFECT PLOT

# EFFECT OF MEAN ON INTERPRETATION: If distribution of the features show that
# they are differnt scale, then the effect of weights will also depend on what
# scale the features are.

# EFFECT OF VARIANCE ON INTERPRETATION: if the variance of a feature is small, then
# they all contribute almost similarly to the prediction (around mean value). But
# it could be that that feature contributes a lot to the prediction despite its

# %%

# lets understand the fatures first %%

# distribution of features

# data = ml_splits_feff["Cu"]
# X = data.train.X
# fig = plt.figure(figsize=(20, 20))
# gs = fig.add_gridspec(len(cfg.compounds), 8, hspace=0, wspace=0)
# axs = gs.subplots(sharey=True, sharex=True)
# for c, ax_row in zip(cfg.compounds, axs):
#     data = ml_splits_feff[c]
#     X = data.train.X
#     for ax, i in zip(ax_row, range(X.shape[1])):
#         # x = np.log(np.abs(X[:, i]))
#         x = X[:, i]
#         ax.hist(x, bins=50)
#         # ax.set_xscale("symlog")
#         # ax.set_title(f"{c} Feature {i}")


# %%


def variance_inflation_factor(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = [
        variance_inflation_factor(X, i) for i in range(X.shape[1])
    ]  # for each feature

    return vif


vifs = {c: variance_inflation_factor(ml_splits[c].train.X) for c in cfg.compounds}


# %%
def plot_vifs(vifs):
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(len(cfg.compounds), 1, hspace=0, wspace=0)
    axs = gs.subplots(sharey=True, sharex=True)
    for c, ax in zip(cfg.compounds, axs):
        ax.set_xlim(0, len(vifs[c]))
        ax.set_yscale("log")
        VIF_LIMIT = 10
        ax.axhline(VIF_LIMIT, color="yellow", linestyle="--", linewidth=1)
        # put text saying number of features with vif > VI_LIMIT
        ax.text(
            0.5,
            0.8,
            f"VIF <= {VIF_LIMIT}: {np.sum(np.array(vifs[c]) <= VIF_LIMIT)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize="xx-large",
        )
        ax.bar(
            range(len(vifs[c])),
            vifs[c],
            color=np.where(np.array(vifs[c]) <= VIF_LIMIT, "blue", "red"),
        )
        ax.legend(fontsize="xx-large")
        # ax.bar(range(len(vifs[c])), vifs[c], label=c)
    axs[-1].set_xlabel("Feature Index")
    for ax in axs.T.flatten():
        ax.set_ylabel("VIF", fontsize="xx-large")
    return axs


plot_vifs(vifs)

# %%


def get_pca_reduced_ml_splits(ml_splits):

    def pca_reduce_split(ml_split):
        pca = PCA(n_components=0.95)
        pca.fit(ml_split.train.X)
        ml_split.train.X = pca.transform(ml_split.train.X)
        ml_split.test.X = pca.transform(ml_split.test.X)
        return ml_split

    return {c: pca_reduce_split(ml_split) for c, ml_split in ml_splits.items()}


from copy import deepcopy

pca_reduced_ml_splits = get_pca_reduced_ml_splits(deepcopy(ml_splits))

# %%
# pca_reduced_models, mses_rel = get_ml_models(model_class, pca_reduced_ml_splits)
pca_reduced_models, mses_rel = get_sm_linear_models(pca_reduced_ml_splits)


# %%

plot_all_feature_weights_types(
    linear_models=pca_reduced_models,
    ml_splits=pca_reduced_ml_splits,
    mses_rel=mses_rel,
    save_file_postfix="pca",
)

# %%

plot_top_n_features(top_n=10, models=ml_models, mses=mses_rel)

# %%


# %%


def get_train_residuals_variance(models, ml_splits):
    return {
        c: (model.predict(ml_splits[c].train.X) - ml_splits[c].train.y).std(axis=0)
        for c, model in models.items()
    }


def plot_variance_of_residuals(resd_var, axs=None):
    if axs is None:
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(4, 2, hspace=0, wspace=0.1)
        axs = gs.subplots()

    for ax, compound in zip(axs.flatten(), cfg.compounds):
        ax.plot(resd_var[compound], label=compound)
        ax.legend()
        ax.set_xlabel("Feature Index", fontsize="xx-large")
        ax.set_ylabel("Std of Train Residuals", fontsize="xx-large")
        ax.set_title(f"Variance of {compound} train residuals", fontsize="xx-large")
    return axs


plot_variance_of_residuals(
    get_train_residuals_variance(
        pca_reduced_models,
        pca_reduced_ml_splits,
    )
)
# %%

# Do weighted linear regression using statsmodels

from statsmodels.regression.linear_model import WLS


def get_wls_models(ml_splits):
    # do simple linear reg to find weights
    models = {
        c: LinearRegression().fit(ml_split.train.X, ml_split.train.y)
        for c, ml_split in ml_splits.items()
    }

    residual_variance = {
        c: (model.predict(ml_split.train.X) - ml_split.train.y).std(axis=0)
        for c, model, ml_split in zip(
            models.keys(), models.values(), ml_splits.values()
        )
    }

    def wls_model(ml_split, residual_variances):
        weights = 1 / residual_variances
        # repeat the weights for each instance for train data
        # weights = (
        weights = np.repeat(weights, ml_split.train.y.shape[0]).reshape(
            ml_split.train.y.shape[0], -1
        )

        wls_models = []
        for i in range(ml_split.train.y.shape[1]):
            wls = WLS(
                ml_split.train.y[:, i],
                # sm.add_constant(ml_split.train.X),
                ml_split.train.X,
                weights=weights[:, i],
            )
            wls_models.append(wls.fit())
        return np.array(wls_models)

    wls_models = {
        c: wls_model(ml_split, residual_variance[c])
        for c, ml_split in ml_splits.items()
    }

    mses_model = {}
    for c, models in wls_models.items():
        mses_model[c] = []
        for i, model in enumerate(models):
            # y_pred = model.predict(sm.add_constant(ml_splits[c].test.X))
            y_pred = model.predict(ml_splits[c].test.X)
            mse = mean_squared_error(ml_splits[c].test.y[:, i], y_pred)
            mses_model[c].append(mse)

    # average mse over all features
    mses_model = {c: np.mean(mses_model[c]) for c in ml_splits.keys()}

    mses_mean = {
        c: mean_squared_error(
            ml_split.test.y,
            np.full_like(ml_split.test.y, np.mean(ml_split.train.y, axis=0)),
        )
        for c, ml_split in ml_splits.items()
    }

    mses_rel = {c: mses_mean[c] / np.array(mses_model[c]) for c in ml_splits.keys()}

    return wls_models, mses_rel


wls_models, mses_rel_wsl = get_wls_models(pca_reduced_ml_splits)
wls_models = {c: DummyLinearModel(wls_models[c]) for c in pca_reduced_ml_splits.keys()}
# %%

plot_significant_weights(
    wls_models,
    mses_rel_wsl,
    pca_reduced_ml_splits,
)

# %%

plot_all_feature_weights_types(
    linear_models=wls_models,
    ml_splits=pca_reduced_ml_splits,
    mses_rel=mses_rel_wsl,
    save_file_postfix="wls",
)

# %%


plot_top_n_features(
    top_n=10,
    models={c: DummyLinearModel(wls_models[c]) for c in cfg.compounds},
    mses=mses_rel_wsl,
)


# %%

# dof fft of the targets in all data

y = ml_splits["Cu"].train.y

for _ in range(10):
    idx = np.random.randint(0, y.shape[0])
    fft = np.abs(np.fft.fft(y[idx]))
    freq = np.fft.fftfreq(len(fft))
    plt.plot(freq, fft)


# %%
