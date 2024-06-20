# %%
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
import numpy as np
import matplotlib.pyplot as plt

from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import LinReg, Trained_FCModel

# %%

# model = LinReg(DataQuery("Cu", "FEFF"))
model_class = Trained_FCModel
model = model_class(DataQuery("Cu", "FEFF"), name="per_compound_tl")
data = load_xas_ml_data(DataQuery("Cu", "FEFF"))

# %%

residues = model.predictions - data.test.y
mean = np.mean(data.test.y, axis=0)
# %%

normalized_residues = residues / np.std(residues, axis=0)
plt.plot(normalized_residues.T)

# %%


std_dev = np.std(data.test.y, axis=0)
plt.scatter(
    data.test.y / std_dev, model.predictions / std_dev, alpha=0.1, s=1, c="black"
)

# %%


def scatter_data(
    compound,
    model_class,
    simulation_type,
    fft=False,
    **model_kwargs,
):
    print("Loading data")
    data = load_xas_ml_data(DataQuery(compound, simulation_type), use_cache=True)
    print("Loaded data")
    model = model_class(DataQuery(compound, simulation_type), **model_kwargs)
    print("Loaded model")
    if fft:
        pred_fft = np.abs(np.fft.fft(model.predictions, axis=0))
        target_fft = np.abs(np.fft.fft(data.test.y, axis=0))
        return target_fft, pred_fft
    std_dev = np.std(data.test.y, axis=0)
    X = data.test.y / std_dev
    y = model.predictions / std_dev
    return X, y


plot_fft = True

# plot avg of fft for predictions and targets
for simulation_type in ["FEFF", "ACSF", "SOAP"]:

    freq = np.fft.fftfreq(
        load_xas_ml_data(DataQuery("Cu", simulation_type)).test.y.shape[1]
    )
    # reduce ffq
    freq = freq[:-2]

    plt.plot(
        freq,
        np.convolve(
            np.mean(
                np.abs(
                    np.fft.fft(
                        LinReg(DataQuery("Cu", simulation_type)).predictions,
                        axis=0,
                    )
                ),
                axis=0,
            ),
            np.ones(3) / 3,
            mode="valid",
        ),
        label=simulation_type,
    )

    # plt.plot(
    #     # freq,
    #     np.mean(
    #         np.abs(
    #             np.fft.fft(
    #                 load_xas_ml_data(DataQuery("Cu", simulation_type)).test.y, axis=0
    #             )
    #         ),
    #         axis=1,
    #     ),
    #     label=simulation_type,
    # )

    # plt.xlim(1, None)
    # plt.ylim(0, 3)
    # # break

# plt.plot(
#     np.convolve(
#         np.mean(
#             np.abs(
#                 np.fft.fft(load_xas_ml_data(DataQuery("Cu", "FEFF")).test.y, axis=0)
#             ),
#             axis=0,
#         ),
#         np.ones(3) / 3,
#         mode="valid",
#     ),
#     label="original",
#     linestyle="--",
# )

plt.legend()

# plt.scatter(
#     *scatter_data("Cu", LinReg, "FEFF", fft=plot_fft),
#     alpha=0.05,
#     s=1,
#     c="black",
# )
# plt.scatter(
#     *scatter_data("Cu", LinReg, "ACSF", fft=plot_fft),
#     alpha=0.05,
#     s=1,
#     c="red",
# )
# plt.scatter(
#     *scatter_data("Cu", LinReg, "SOAP", fft=plot_fft),
#     alpha=0.05,
#     s=1,
#     c="blue",
# )
# plt.legend(
#     handles=[
#         plt.Line2D(
#             [0], [0], marker="o", color="w", markerfacecolor="black", markersize=5
#         ),
#         plt.Line2D(
#             [0], [0], marker="o", color="w", markerfacecolor="red", markersize=5
#         ),
#         plt.Line2D(
#             [0], [0], marker="o", color="w", markerfacecolor="blue", markersize=5
#         ),
#     ],
#     labels=["FEFF", "ACSF", "SOAP"],
# )
# plt.xscale("log")
# plt.yscale("log")

# %%

# plt.plot(
#     np.mean(
#         load_xas_ml_data(DataQuery("Cu", "FEFF")).test.y,
#         axis=0,
#     )
# )
plt.plot(
    np.mean(
        LinReg(DataQuery("Cu", "FEFF")).predictions
        - load_xas_ml_data(DataQuery("Cu", "FEFF")).test.y,
        axis=0,
    ),
    label="FEFF",
)
plt.plot(
    np.mean(
        LinReg(DataQuery("Cu", "ACSF")).predictions
        - load_xas_ml_data(DataQuery("Cu", "ACSF")).test.y,
        axis=0,
    ),
    label="ACSF",
)
plt.plot(
    np.mean(
        LinReg(DataQuery("Cu", "SOAP")).predictions
        - load_xas_ml_data(DataQuery("Cu", "SOAP")).test.y,
        axis=0,
    ),
    label="SOAP",
)
ax2 = plt.gca().twinx()
ax2.plot(
    np.mean(
        load_xas_ml_data(DataQuery("Cu", "FEFF")).test.y,
        axis=0,
    ),
    label="FEFF",
    linestyle="--",
)
plt.plot
plt.legend()


# %%
