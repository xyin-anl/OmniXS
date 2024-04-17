# %%
import os

os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from config.defaults import cfg
from src.data.ml_data import DataQuery, load_xas_ml_data

# %%


# simulation_type = "FEFF"
# compounds = cfg.compounds

# type = "PCA"
type = "TSNE"
simulation_type = "VASP"
compounds = ["Cu", "Ti"]

axs_dims = (len(compounds), 3)
axes_size = 4
fig, axs = plt.subplots(
    axs_dims[0],
    axs_dims[1],
    figsize=(axs_dims[1] * axes_size, axs_dims[0] * axes_size),
    sharex=True,
    sharey=True,
)
for ax_pair, compound in zip(axs, compounds):
    query = DataQuery(compound=compound, simulation_type=simulation_type)
    data_splits = load_xas_ml_data(
        query=query,
        split_fractions=(0.8, 0.1, 0.1),
        pca_with_std_scaling=None,
        scale_feature_and_target=False,  # it will be scaled later in code
        filter_spectra_anomalies=True,
    )

    # merge all data
    X = np.concatenate(
        [data_splits.train.X, data_splits.val.X, data_splits.test.X],
    )
    y = np.concatenate(
        [data_splits.train.y, data_splits.val.y, data_splits.test.y],
    )

    # colored by clustering within the data dims

    max_dim = 2
    # use tsne

    if type == "TSNE":
        X_embedding = manifold.TSNE(n_components=max_dim).fit_transform(
            StandardScaler().fit_transform(X)
        )
    elif type == "PCA":
        X_embedding = PCA(n_components=max_dim).fit_transform(
            StandardScaler().fit_transform(X)
        )

    ax = ax_pair[0]
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    ax.scatter(
        x=X_embedding[:, 0],
        y=X_embedding[:, 1],
        c=kmeans.labels_,
        cmap="jet",
        s=1,
    )

    ax = ax_pair[1]
    # colored by kmean clustering of y values
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    ax.scatter(
        x=X_embedding[:, 0],
        y=X_embedding[:, 1],
        c=kmeans.labels_,
        cmap="jet",
        s=1,
    )

    ax = ax_pair[2]
    model = LinearRegression().fit(X_embedding, y)
    y_pred = model.predict(X_embedding)
    y_residue = y - y_pred
    y_rmse = np.sqrt(np.mean(y_residue**2, axis=1))
    ax.scatter(
        x=X_embedding[:, 0],
        y=X_embedding[:, 1],
        c=LogNorm()(y_rmse),
        s=1,
        cmap="jet",
    )

    # break

for row, c in zip(axs, compounds):
    row[0].set_ylabel(f"{c}", fontsize=18)

for col in axs:
    font_size = 18
    col[0].set_title("Feature clustering", fontsize=font_size)
    col[1].set_title("Target clustering", fontsize=font_size)
    col[2].set_title("RMSE of linear fit", fontsize=font_size)
    break

for ax in axs[-1]:
    xlabel = "PCA dim 2" if type == "PCA" else "TSNE dim 2"
    ax.set_xlabel(xlabel, fontsize=18)

for ax in axs.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
file_name = (
    f"pcas_{simulation_type}.pdf" if type == "PCA" else f"tsnes_{simulation_type}.pdf"
)
plt.savefig(file_name, dpi=300)
plt.show()

# %%
