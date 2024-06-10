# %%

import pickle
from copy import deepcopy
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
from p_tqdm import p_map

from config.defaults import cfg
from src.analysis.plots import Plot
from src.data.ml_data import DataQuery, load_xas_ml_data
from src.models.trained_models import ElastNet, MeanModel, Trained_FCModel, XGBReg

# %%


def plot_deciles_of_top_predictions(
    models: Dict[str, Callable] = None,  # default is fc_tl for all compounds
    simulation_type="FEFF",
    splits=10,
):

    if models is None:
        models = {
            c: Trained_FCModel(DataQuery(c, simulation_type), name=f"{c}_tl")
            for c in cfg.compounds
        }
    compounds = models.keys()

    model_name = set([model.name for model in models.values()])
    model_name = None if len(model_name) > 1 else model_name.pop()

    fig, axs = plt.subplots(
        splits,
        len(compounds),
        figsize=(20, 16),
        sharex=True,
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    for c, axs in zip(compounds, axs.T):
        Plot().set_title(f"{c}").plot_top_predictions(
            models[c].top_predictions(splits=splits),
            splits=splits,
            axs=axs,
            compound=c,
            color_background=True,
        )

        title = f"{models[c].name} {simulation_type}"
        title += f"\nrel MSE: {models[c].mse_relative_to_mean_model:.2f}"
        axs[0].set_title(title)

    plt.tight_layout()

    plt.savefig(
        f"top_predictions_{model_name}_{simulation_type}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.savefig(
        f"top_predictions_{model_name}_{simulation_type}.png",
        dpi=300,
        bbox_inches="tight",
    )

    for c in compounds:
        with open(f"{model_name}_{c}_{simulation_type}.pkl", "wb") as f:
            pickle.dump(models[c], f)

    return axs


# %%


model_name = "ft_tl"  # "per_compound_tl"
plot_deciles_of_top_predictions(
    models={
        c: Trained_FCModel(DataQuery(c, "FEFF"), name=model_name) for c in cfg.compounds
    },
    simulation_type="FEFF",
)

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
        XGBReg,
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
        XGBReg,
        # lambda q: Trained_FCModel(q, name="ft_tl"),
        # lambda q: Trained_FCModel(q, name="per_compound_tl"),
    ],
)

# %%

load_xas_ml_data(DataQuery("Mn", "SOAP")).train.X.shape
