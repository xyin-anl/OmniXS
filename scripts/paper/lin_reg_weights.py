# %%

from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import LinReg, Trained_FCModel
from config.defaults import cfg
from matplotlib import pyplot as plt
import numpy as np

# %%


weights = {
    c: LinReg(DataQuery(c, "FEFF"), name="per_compound_tl").model.coef_
    for c in cfg.compounds
}

# %%

fig, axs = plt.subplots(4, 2, figsize=(20, 8))
for ax, c in zip(axs.flatten(), cfg.compounds):
    ax.imshow(weights[c], aspect="auto")

# %%

# pca of weights
from sklearn.decomposition import PCA

pca_weights = {c: PCA(n_components=2).fit_transform(weights[c]) for c in cfg.compounds}

# %%

fig, axs = plt.subplots(4, 2, figsize=(8, 8))
for ax, c in zip(axs.flatten(), cfg.compounds):
    ax.plot(pca_weights[c][:, 0], pca_weights[c][:, 1])
    ax.set_title(c)

# %%

# umap of thw weights

import umap

umap_weights = {c: umap.UMAP().fit_transform(weights[c]) for c in cfg.compounds}

# %%

fig, axs = plt.subplots(4, 2, figsize=(8, 8))
for ax, c in zip(axs.flatten(), cfg.compounds):
    ax.plot(umap_weights[c][:, 0], umap_weights[c][:, 1])
    ax.set_title(c)

# %%

pca3_weights = {c: PCA(n_components=3).fit_transform(weights[c]) for c in cfg.compounds}

# %%

fig, axs = plt.subplots(4, 2, figsize=(8, 8), subplot_kw={"projection": "3d"})
for ax, c in zip(axs.flatten(), cfg.compounds):
    ax.scatter(*pca3_weights[c].T)
    ax.set_title(c)
# %%

umap3_weights = {
    c: umap.UMAP(n_components=3).fit_transform(weights[c]) for c in cfg.compounds
}

# %%

fig, axs = plt.subplots(4, 2, figsize=(8, 8), subplot_kw={"projection": "3d"})
for ax, c in zip(axs.flatten(), cfg.compounds):
    ax.scatter(*umap3_weights[c].T)
    ax.set_title(c)

# %%

data = {c: load_xas_ml_data(DataQuery(c, "FEFF")) for c in cfg.compounds}

# %%

x = np.concatenate([data[c].train.X for c in cfg.compounds])
y = np.concatenate([data[c].train.y for c in cfg.compounds])


from sklearn.linear_model import LinearRegression

lr_univ = LinearRegression().fit(x, y)
lr_univ.coef_

pca_lr = PCA(n_components=2).fit_transform(lr_univ.coef_)
umap_lr = umap.UMAP().fit_transform(lr_univ.coef_)

# plt.plot(pca_lr[:, 0], pca_lr[:, 1])
plt.plot(umap_lr[:, 0], umap_lr[:, 1])

# %%
