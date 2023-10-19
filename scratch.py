# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# %%

compound = "Ti-O"
simulation_type = "VASP"  # "VASP", "FEFF"
dir_name = (
    f"dataset/ML-231009/{compound}_K-edge_{simulation_type}_XANES/material-splits/data/"
)
save_file_prefix = f"{compound}_{simulation_type}"

selected_x_pca_dims = {"FEFF": {"Cu-O": 2, "Ti-O": 4}, "VASP": {"Ti-O": 2}}
selected_x_pca_dim = selected_x_pca_dims[simulation_type][compound]

X_train = np.load(dir_name + "X_train.npy")
X_test = np.load(dir_name + "X_test.npy")
y_train = np.load(dir_name + "y_train.npy")
y_test = np.load(dir_name + "y_test.npy")

# %%

max_dim = 5
x_pca = PCA(n_components=max_dim).fit_transform(X_train)
y_pca = PCA(n_components=1).fit_transform(y_train)

# %%

plt.figure(figsize=(20, 20))  # Increase figure size
pairs = list(combinations_with_replacement(range(max_dim), 2))

for i in range(max_dim):
    for j in range(i + 1, max_dim):
        plt.subplot(max_dim, max_dim, i * max_dim + j + 1)
        plt.scatter(x_pca[:, i], x_pca[:, j], c=y_pca, cmap="viridis", marker=".")
        plt.title(f"x_pca_dim = {i}, {j}")
# plt.suptitle(f"{compound} {simulation_type}")
plt.tight_layout()
plt.savefig(f"{save_file_prefix}_x_pca_all.pdf", dpi=300)

# %%


xy_pca = np.stack([x_pca[:, selected_x_pca_dim], y_pca[:, 0]], axis=1)
xy_pca_sorted = np.sort(xy_pca, axis=0)
plt.scatter(
    xy_pca_sorted[:, 0],
    xy_pca_sorted[:, 1],
    marker=".",
    facecolors="none",
    edgecolors="blue",
    alpha=0.5,
)
m = LinearRegression().fit(xy_pca_sorted[:, 0].reshape(-1, 1), xy_pca_sorted[:, 1])
slope = m.coef_[0]
intercept = m.intercept_
r2 = m.score(xy_pca_sorted[:, 0].reshape(-1, 1), xy_pca_sorted[:, 1])
plt.plot(
    xy_pca_sorted[:, 0],
    xy_pca_sorted[:, 0] * slope + intercept,
    c="red",
    label=f"R^2 = {round(r2,2)}",
)

plt.xlabel(f"x_pca_dim = {selected_x_pca_dim}")
plt.ylabel("y_pca_dim = 0")
plt.title(f"{compound} {simulation_type}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_file_prefix}_x_pca_selected_fit_{selected_x_pca_dim}.pdf", dpi=300)
plt.show()

# %%
plt.scatter(
    x_pca[:, 0],
    x_pca[:, selected_x_pca_dim],
    c=y_pca,
    marker=".",
    cmap="viridis",
)
plt.xlabel("x_pca_dim = 0")
plt.ylabel(f"x_pca_dim = {selected_x_pca_dim}")
plt.title(f"{compound} {simulation_type}")
plt.savefig(
    f"{save_file_prefix}_x_pca_selected_{selected_x_pca_dim}_scatter.pdf", dpi=300
)
plt.show()


# %%

linear_model = LinearRegression().fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
linreg_rmse = np.sqrt(np.mean((((y_test - y_pred) ** 2)).flatten()))

# %%
model = linear_model
y_residue = y_test - y_pred
y_mean = y_test.mean(axis=0)
plt.figure(figsize=(10, 8))
for percentile in np.arange(10, 100, 10):
    y_residue = y_pred - y_test
    y_percentile = np.percentile(y_residue, percentile, axis=0)
    plt.plot(y_mean + y_percentile, alpha=0.5, linestyle="--", label=f"{percentile}%")
    # plt.plot( y_percentile, alpha=0.5)
# plt.ylim((0.85, 1.45))
plt.plot(y_mean, c="red", label="y_mean")
plt.legend(title="residue_percentile")
plt.xlabel("index")
plt.ylabel("y_mean + residue_percentile")
plt.title(f"{compound} {simulation_type}")
plt.savefig(f"{save_file_prefix}_residue_percentile.pdf", dpi=300)
plt.show()


# %%
heatmap_of_lines(y_test, ylabel="y_test")
plt.title(f"Histogram of test spectra: {compound} {simulation_type}")
plt.savefig(f"{save_file_prefix}_heatmap_y_test.pdf", dpi=300)
plt.show()
# %%

heatmap_of_lines(y_pred, ylabel="y_pred")
plot_title = f"Histogram of Linear Regressor predictions: {compound} {simulation_type}"
plot_title += f"\n RMSE = {round(linreg_rmse, 3)}"
plt.title(plot_title)
plt.savefig(f"{save_file_prefix}_heatmap_y_pred.pdf", dpi=300)
plt.show()

# %%
heatmap_of_lines(y_pred - y_test, ylabel="y_pred - y_test")
plot_title = f"Histogram of Linear Regressor residue: {compound} {simulation_type}"
plot_title += f"\n RMSE = {round(linreg_rmse, 3)}"
plt.title(plot_title)
plt.savefig(f"{save_file_prefix}_heatmap_residue.pdf", dpi=300)
plt.show()

# %%
