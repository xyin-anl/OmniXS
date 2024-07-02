# %%


from config.defaults import cfg
import scipy.stats
from src.data.ml_data import load_xas_ml_data, load_all_data, DataQuery
import numpy as np
from matplotlib import pyplot as plt

# %%


def kl_divergence(compound1, compound2, epsilon=1e-10, bins=100):
    data1 = load_xas_ml_data(DataQuery(compound1, "FEFF"))
    data2 = load_xas_ml_data(DataQuery(compound2, "FEFF"))
    # take mean of spectras
    # spectra1 = data1.train.X.mean(axis=0)
    # spectra2 = data2.train.X.mean(axis=0)
    spectra1 = data1.train.X.flatten()
    spectra2 = data2.train.X.flatten()
    min_bins = min(spectra1.min(), spectra2.min())
    max_bins = max(spectra1.max(), spectra2.max())
    common_bins = np.linspace(min_bins, max_bins, bins)
    p_hist, _ = np.histogram(spectra1, bins=common_bins, density=True)
    q_hist, _ = np.histogram(spectra2, bins=common_bins, density=True)
    return scipy.stats.entropy(p_hist + epsilon, q_hist + epsilon)


compounds = cfg.compounds

kl_matrix = np.zeros((len(compounds), len(compounds)))
for i, compound1 in enumerate(compounds):
    for j, compound2 in enumerate(compounds):
        kl_matrix[i, j] = kl_divergence(compound1, compound2)

# %%

# larger fig
plt.figure(figsize=(8, 6))
plt.style.use(["default", "science", "vibrant"])
plt.imshow(kl_matrix)
plt.colorbar()
plt.xticks(range(len(compounds)), compounds, rotation=90, fontsize=15)
plt.yticks(range(len(compounds)), compounds, fontsize=15)
plt.title("KL Divergence between compounds", fontsize=20)
plt.show()

# %%
import math
import numpy as np


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(
        np.dot(H, K), H
    )  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


# if __name__ == "__main__":
#     X = np.random.randn(100, 64)
#     Y = np.random.randn(100, 64)

#     print("Linear CKA, between X and Y: {}".format(linear_CKA(X, Y)))
#     print("Linear CKA, between X and X: {}".format(linear_CKA(X, X)))

#     print("RBF Kernel CKA, between X and Y: {}".format(kernel_CKA(X, Y)))
#     print("RBF Kernel CKA, between X and X: {}".format(kernel_CKA(X, X)))


# %%

feature_matrix = np.zeros((len(compounds), len(compounds)))
for i, compound1 in enumerate(compounds):
    for j, compound2 in enumerate(compounds):
        data1 = load_xas_ml_data(DataQuery(compound1, "FEFF"))
        data2 = load_xas_ml_data(DataQuery(compound2, "FEFF"))
        X1 = data1.train.X
        X2 = data2.train.X
        X1 = X1.mean(axis=0).reshape(-1, 1)
        X2 = X2.mean(axis=0).reshape(-1, 1)
        feature_matrix[i, j] = kernel_CKA(X1, X2)


# %%

plt.imshow(feature_matrix)
# add legend
plt.xticks(range(len(compounds)), compounds, rotation=90)
plt.yticks(range(len(compounds)), compounds)
plt.title("CKA between compounds")
plt.colorbar()

# %%

from src.data.ml_data import load_xas_ml_data, load_all_data, DataQuery

compounds = cfg.compounds
cka_with_all = np.zeros(len(compounds))
for i, compound in enumerate(compounds):
    data = load_xas_ml_data(DataQuery(compound, "FEFF"))
    # data_rest = load_all_data("FEFF", compounds=[x for x in compounds if x != compound])
    data_rest = load_all_data("FEFF")
    X1 = data.train.X
    X1 = X1.mean(axis=0).reshape(-1, 1)
    X2 = data_rest.train.X.mean(axis=0).reshape(-1, 1)
    cka_with_all[i] = kernel_CKA(X1, X2)


# %%

cke_with_all_others = cka_with_all
# larger fig
plt.figure(figsize=(8, 6))
plt.style.use(["default", "science", "vibrant"])
plt.bar(compounds, cke_with_all_others)
plt.xticks(rotation=90, fontsize=15)
plt.title("CKA with all other compounds", fontsize=20)
plt.show()






# %%
# cka acroos compound

from src.data.ml_data import load_xas_ml_data, load_all_data, DataQuery
from src.models.trained_models import Trained_FCModel
from config.defaults import cfg
import numpy as np
import matplotlib.pyplot as plt

compounds = cfg.compounds
cka_across_compounds = np.zeros((len(compounds), len(compounds)))

for i, compound1 in enumerate(compounds):
    for j, compound2 in enumerate(compounds):
        data1 = load_xas_ml_data(DataQuery(compound1, "FEFF"))
        data2 = load_xas_ml_data(DataQuery(compound2, "FEFF"))
        X1 = data1.train.X
        X1 = X1.mean(axis=0).reshape(-1, 1)
        X2 = data2.train.X
        X2 = X2.mean(axis=0).reshape(-1, 1)
        cka_across_compounds[i, j] = kernel_CKA(X1, X2)

# %%

plt.figure(figsize=(8, 6))
plt.style.use(["default", "science", "vibrant"])
plt.imshow(cka_across_compounds)
plt.colorbar()
plt.xticks(range(len(compounds)), compounds, rotation=90, fontsize=15)
plt.yticks(range(len(compounds)), compounds, fontsize=15)
plt.title("CKA across compounds", fontsize=20)
plt.show()


# %%


np.save("cka_witH_all.npy", cka_with_all)

# %%
from src.data.ml_data import load_all_data
from src.data.ml_data import MLSplits, DataSplit
from src.models.trained_models import Trained_FCModel
from src.data.ml_data import DataQuery, load_xas_ml_data
from scripts.paper.plot_utils import Plot
from config.defaults import cfg
import numpy as np
import matplotlib.pyplot as plt
from functools import cached_property

# %%

model_names = ["universal_tl", "ft_tl"]

mse_before = [
    Trained_FCModel(DataQuery(c, "FEFF"), name="per_compound_tl").mse
    for c in cfg.compounds
]
mse_after = [
    Trained_FCModel(DataQuery(c, "FEFF"), name="ft_tl").mse for c in cfg.compounds
]
mse_diff = np.array(mse_after) - np.array(mse_before)

# %%

plt.figure(figsize=(8, 6))
# color all Cr, Ti, Cu and Ni red
colors = ["red" if x in ["Cu", "Ni", "Ti", "Cr"] else "blue" for x in cfg.compounds]
plt.style.use(["default", "science", "vibrant"])
plt.scatter(mse_diff, cka_with_all, s=300, c=colors)
plt.xlabel("MSE difference after fine-tuning", fontsize=20)
plt.ylabel("CKA with all compounds", fontsize=20)
plt.title("MSE difference vs CKA with all compounds", fontsize=20)
# add text labels centered 
for i, txt in enumerate(cfg.compounds):
    plt.annotate(txt, (mse_diff[i]-0.0002, cka_with_all[i]), ha="center", va="center", fontsize=13)
## add add fit to the plot using slotpe and interecept
plt.plot(np.unique(mse_diff), np.poly1d(np.polyfit(mse_diff, cka_with_all, 1))(np.unique(mse_diff)), color="red", linewidth=2)






# plt.scatter(mse_diff, cka_with_all)
# plt.xlabel("MSE difference after fine-tuning")
# plt.ylabel("CKA with all compounds")
# plt.title("MSE difference vs CKA with all compounds")
# plt.show()


# %%

bar_width = 0.35
fig, ax = plt.subplots()
index = np.arange(len(cfg.compounds))
bar1 = ax.bar(index, mse_before, bar_width, label="Before fine-tuning")
bar2 = ax.bar(index + bar_width, mse_after, bar_width, label="After fine-tuning")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(cfg.compounds, rotation=90)
ax.legend()
plt.show()

# plt.bar(cfg.compounds, -mse_diff)
# plt.xticks(rotation=90)
# plt.title("MSE difference after fine-tuning")
# plt.show()

# plt.bar(compounds, cka_with_all)
# plt.xticks(rotation=90)
# plt.title("CKA with all compounds")
# plt.show()

# %%

# cke  and mse diff

bar_width = 0.35
fig, ax = plt.subplots()
index = np.arange(len(cfg.compounds))
bar1 = ax.bar(index, -mse_diff, bar_width, label="MSE difference after fine-tuning")
bar2 = ax.bar(
    index + bar_width, cka_with_all / 500, bar_width, label="CKA with all compounds"
)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(cfg.compounds, rotation=90)
ax.legend()

# %%

# do scatter plot
plt.scatter(mse_diff, cka_with_all)
plt.xlabel("MSE difference after fine-tuning")
plt.ylabel("CKA with all compounds")
plt.title("MSE difference vs CKA with all compounds")
# add text labels
for i, txt in enumerate(cfg.compounds):
    plt.annotate(txt, (mse_diff[i], cka_with_all[i]))
plt.show()
# %%
# make color of Cu, Ni, Ti, Cu red
colors = ["red" if x in ["Cu", "Ni", "Ti", "Cr"] else "blue" for x in cfg.compounds]
plt.scatter(mse_diff, cka_with_all, c=colors)
plt.xlabel("MSE difference after fine-tuning")
plt.ylabel("CKA with all compounds")
plt.title("MSE difference vs CKA with all compounds")
# add text labels
for i, txt in enumerate(cfg.compounds):
    plt.annotate(txt, (mse_diff[i], cka_with_all[i]))
plt.show()

data_sizes = [
    len(load_xas_ml_data(DataQuery(c, "FEFF")).train.y) for c in cfg.compounds
]
data_sizes_normalzed = np.array(data_sizes) / np.sum(data_sizes)
# %%

# now make the plot size propotional to the data size
# colors = ["red" if x in ["Cu", "Ni", "Ti", "Cr"] else "blue" for x in cfg.compounds]
# color by grandent with data size
import scienceplots

plt.style.use(["default", "science", "vibrant"])
plt.figure(figsize=(8, 6))
colors = data_sizes_normalzed
# scale by log
colors = np.log(data_sizes_normalzed)
colors = (colors - colors.min()) / (colors.max() - colors.min())
plt.scatter(mse_diff, cka_with_all, s=data_sizes_normalzed * 4000, c=colors)
plt.xlabel("MSE difference after fine-tuning", fontsize=20)
plt.ylabel("CKA with all compounds", fontsize=20)
plt.title("Learning form all FEFF", fontsize=20)
# set text centered larger than the point
for i, txt in enumerate(cfg.compounds):
    plt.annotate(
        txt,
        (mse_diff[i], cka_with_all[i]),
        ha="center",
        va="center",
        fontsize=13,
        # use contrastive color
        color="black" if colors[i] > 0.5 else "white",
    )
# show daata size legend
# add tile to color bar
# plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), label="Normalized data size", ax=plt.gca())
# larger title with rotation of 180

# plt.colorbar(
#     plt.cm.ScalarMappable(cmap="viridis"),
#     label="Normalized data size",
#     ax=plt.gca(),
#     orientation="horizontal",
#     pad=0.2,
#     # aspect=40,
#     shrink=0.6,
# )

plt.tight_layout()


# %%

data_all = load_all_data("FEFF").train.X.flatten()
data = [load_xas_ml_data(DataQuery(c, "FEFF")).train.X.flatten() for c in cfg.compounds]

# %%

bin_count = 20
plt.figure(figsize=(8, 6))
plt.style.use(["default", "science", "vibrant"])
common_bins = np.linspace(data_all.min(), data_all.max(), bin_count)
plt.hist(data, bins=common_bins, label=cfg.compounds, alpha=0.5)

# %%
# plot the fit kernel density
from scipy.stats import gaussian_kde

bin_count = 500
common_bins = np.linspace(data_all.min(), data_all.max(), bin_count)
plt.figure(figsize=(8, 6))
for i, d in enumerate(data):
    kde = gaussian_kde(d)
    plt.plot(common_bins, kde(common_bins), label=cfg.compounds[i])
plt.plot(
    common_bins,
    gaussian_kde(data_all)(common_bins),
    label="All",
    color="black",
    linewidth=2,
)
plt.style.use(["default", "science", "vibrant"])
plt.xlabel("M3GNet Latent Space Value", fontsize=20)
# make x sacle wider
plt.xlim(-2000, 2000)
plt.ylabel("Density", fontsize=20)
plt.title("Density of M3GNet Latent Space Values", fontsize=20)
plt.legend(fontsize=15)

# %%

# kl divergence with data all for all compoujnds

bins = 100
data_all = load_all_data("FEFF")
kl_divergence_with_all = np.zeros(len(cfg.compounds))
for i, compound in enumerate(cfg.compounds):
    data = load_xas_ml_data(DataQuery(compound, "FEFF"))
    X1 = data.train.X.flatten()
    X2 = data_all.train.X.flatten()
    min_bins = min(X1.min(), X2.min())
    max_bins = max(X1.max(), X2.max())
    common_bins = np.linspace(min_bins, max_bins, bins)
    p_hist, _ = np.histogram(X1, bins=common_bins, density=True)
    q_hist, _ = np.histogram(X2, bins=common_bins, density=True)
    kl_divergence_with_all[i] = scipy.stats.entropy(p_hist + 1e-10, q_hist + 1e-10)

# %%

# sctterplot of kl_divergence_with_all and cka_with_all

plt.figure(figsize=(8, 6))
plt.style.use(["default", "science", "vibrant"])
plt.scatter(
    kl_divergence_with_all, cka_with_all, s=data_sizes_normalzed * 5000, c=colors
)
plt.xlabel("KL Divergence with all compounds", fontsize=20)
plt.ylabel("CKA with all compounds", fontsize=20)
plt.title("Relation between latent space similarity metrics", fontsize=20)

# plt.colorbar(
#     plt.cm.ScalarMappable(cmap="viridis"),
#     label="Normalized data size",
#     ax=plt.gca(),
#     orientation="horizontal",
#     pad=0.2,
#     shrink=0.6,
# )

# aad text
for i, txt in enumerate(cfg.compounds):
    plt.annotate(
        txt,
        (kl_divergence_with_all[i], cka_with_all[i]),
        ha="center",
        va="center",
        fontsize=13,
        color="black" if colors[i] > 0.5 else "white",
    )
plt.tight_layout()


# %%

# kl divergence vs mse diff
plt.figure(figsize=(8, 6))
plt.style.use(["default", "science", "vibrant"])
plt.scatter(mse_diff, kl_divergence_with_all, s=data_sizes_normalzed * 5000, c=colors)
plt.xlabel("MSE difference after fine-tuning", fontsize=20)
plt.ylabel("KL Divergence with all compounds", fontsize=20)
plt.title("Learning from all FEFF", fontsize=20)
for i, txt in enumerate(cfg.compounds):
    plt.annotate(
        txt,
        (mse_diff[i], kl_divergence_with_all[i]),
        ha="center",
        va="center",
        fontsize=13,
        color="black" if colors[i] > 0.5 else "white",
    )
plt.tight_layout()

# %%

# densit plot of cke of compounds with each other


fro
