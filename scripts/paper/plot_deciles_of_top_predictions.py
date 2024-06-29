# %%

from itertools import product
import scienceplots
import pickle
from copy import deepcopy
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
from p_tqdm import p_map

from config.defaults import cfg
from src.analysis.plots import Plot
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import ElastNet, MeanModel, Trained_FCModel, XGBReg


def plot_deciles_of_top_predictions(
    models: Dict[str, Callable] = None,  # default is fc_tl for all compounds
    simulation_type="FEFF",
    splits=10,
    axs=None,
):

    if models is None:
        models = {
            c: Trained_FCModel(DataQuery(c, simulation_type), name=f"{c}_tl")
            for c in cfg.compounds
        }
    compounds = models.keys()

    model_name = set([model.name for model in models.values()])
    model_name = None if len(model_name) > 1 else model_name.pop()

    if axs is None:
        fig, axs = plt.subplots(
            splits - 1,
            len(compounds),
            figsize=(20, 16),
            sharex=True,
            gridspec_kw={"hspace": 0, "wspace": 0},
        )

    cmap = "tab10"
    colors = {c: plt.get_cmap(cmap)(i) for i, c in enumerate(compounds)}

    for c, axs in zip(compounds, axs.T):
        Plot().set_title(f"{c}", fontsize=FONTSIZE * 0.6).plot_top_predictions(
            models[c].top_predictions(splits=splits),
            splits=splits,
            axs=axs,
            compound=c,
            color_background=False,
            color=colors[c],
        )

        # title = f"{models[c].name} {simulation_type}"
        # title += f"\nrel MSE: {models[c].mse_relative_to_mean_model:.2f}"
        # axs[0].set_title(title)

        for ax in axs:
            # no frame below
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            # no background color
            ax.set_facecolor("white")
            ax.set_title("")
        # axs[0].set_title(f"{c}")
        # add tax on the top with bbox with same background color

        axs[0].set_title(
            f"{c} (" + r"$\bf{\eta=}$" + f"{models[c].mse_relative_to_mean_model:.1f})",
            loc="center",
            fontsize=16,
            # bold
            # fontweight="bold",
            color=colors[c],
            y=1.025,
            x=0.5,
        )

        # axs[0].title.set_bbox(
        #     dict(
        #         # facecolor with alpha values added to the original color
        #         # facecolor=colors[c] + (0.5,), # ERROR:  RGBA sequence should have length 3 or 4
        #         facecolor="white",
        #         edgecolor=colors[c],
        #     ),
        # )

    # plt.savefig(
    #     f"top_predictions_{model_name}_{simulation_type}.pdf",
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    # plt.savefig(
    #     f"top_predictions_{model_name}_{simulation_type}.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    for c in compounds:
        with open(f"{model_name}_{c}_{simulation_type}.pkl", "wb") as f:
            pickle.dump(models[c], f)

    return axs


model_name = "ft_tl"  # "per_compound_tl"
fig, axs = plt.subplots(
    9,
    3,
    # figsize=(20, 16),
    figsize=(8, 12),
    sharex=True,
    gridspec_kw={"hspace": 0, "wspace": 0},
    # gridspec_kw={"hspace": -0.3, "wspace": 0},
)

# for ax in axs.flat:  # hspace is -ve for space saving
#     ax.patch.set_alpha(0.0)
FONTSIZE = 18
for i, ax in enumerate(axs[:, 0], start=1):
    ax.set_ylabel(
        # f"D{i}",
        r"$\bf{D}_{" + f"{i}" + r"}$",
        rotation=0,
        fontsize=FONTSIZE * 0.4,
        labelpad=-6,
        loc="center",
        alpha=1,
        color="gray",
    )

# for ax in axs.flat:
#     ax.patch.set_facecolor("none")  #

plt.style.use(["default", "science"])
simulation_type = "FEFF"


plot_deciles_of_top_predictions(
    models={
        c: Trained_FCModel(DataQuery(c, simulation_type), name=model_name)
        for c in ["Mn", "Fe", "Cu"]
    },
    simulation_type="FEFF",
    axs=axs,
)


fig.savefig(
    f"top_predictions_{model_name}_{simulation_type}_{axs.shape[1]}_compounds.pdf",
    dpi=300,
    bbox_inches="tight",
)


# %%


model_name = "ft_tl"  # "per_compound_tl"
plt.style.use(["default", "science"])
plot_deciles_of_top_predictions(
    models={
        c: Trained_FCModel(DataQuery(c, "FEFF"), name=model_name) for c in cfg.compounds
    },
    simulation_type="FEFF",
)

# %%

# %%

mses = {
    c: Trained_FCModel(DataQuery(c, "FEFF"), name="ft_tl").mse_relative_to_mean_model
    for c in cfg.compounds
}
# find compound corresponding to min, max, median
min_c, max_c, median_c = (
    sorted(mses, key=mses.get)[0],
    sorted(mses, key=mses.get)[-1],
    sorted(mses, key=mses.get)[len(mses) // 2],
)
min_c, max_c, median_c  # ("Cu", "Mn", "Fe")


# %%

# # FOR UNIVERSAL model the data has to be changed for each model
# univ_model = Trained_FCModel(DataQuery("ALL", "FEFF"), name="universal_tl")
# models = {}
# for c in cfg.compounds:
#     univ_model.data = load_xas_ml_data(DataQuery(c, "FEFF"))
#     models[c] = deepcopy(univ_model)
# axs = plot_deciles_of_top_predictions(models=models, simulation_type="FEFF")

# %%

# plot all specialized models
simulation_type = "ACSF"
p_map(
    lambda model_class: plot_deciles_of_top_predictions(
        models={c: model_class(DataQuery(c, simulation_type)) for c in cfg.compounds},
        simulation_type=simulation_type,
    ),
    [
        # MeanModel,
        ElastNet,
        # XGBReg,
        # lambda q: Trained_FCModel(q, name="ft_tl"),
        # lambda q: Trained_FCModel(q, name="per_compound_tl"),
    ],
)

simulation_type = "SOAP"
p_map(
    lambda model_class: plot_deciles_of_top_predictions(
        models={c: model_class(DataQuery(c, simulation_type)) for c in cfg.compounds},
        simulation_type=simulation_type,
    ),
    [
        # MeanModel,
        ElastNet,
        # XGBReg,
        # lambda q: Trained_FCModel(q, name="ft_tl"),
        # lambda q: Trained_FCModel(q, name="per_compound_tl"),
    ],
)

# %%
