# %%
import random
from utils.src.plots.heatmap_of_lines import heatmap_of_lines
import numpy as np
from config.defaults import cfg
from scripts.plots.plot_all_spectras import MLDATAPlotter
import matplotlib.pyplot as plt
from src.data.ml_data import load_all_data, load_xas_ml_data, DataQuery
from src.data.vasp_data_raw import RAWDataVASP
from src.data.vasp_data import VASPData
from src.data.vasp_data_raw import RAWDataVASP
from src.data.vasp_data import VASPData
from scripts.model_scripts.plot_universal_tl_model import (
    plot_universal_tl_vs_per_compound_tl,
    universal_TL_mses,
)

# %%

data_sizes = {
    c: len(
        load_xas_ml_data(
            DataQuery(compound=c, simulation_type="FEFF"),
        ).train.X
    )
    for c in cfg.compounds
}

# %%

data_fraction = {c: data_sizes[c] / sum(data_sizes.values()) for c in cfg.compounds}

# %%

from src.models.trained_models import Trained_FCModel

per_compound_tl_mse = {
    c: Trained_FCModel(
        query=DataQuery(compound=c, simulation_type="FEFF"),
        name="per_compound_tl",
    ).mse
    for c in cfg.compounds
}

# %%

per_compound_ft_tl_mse = {
    c: Trained_FCModel(
        query=DataQuery(compound=c, simulation_type="FEFF"),
        name="ft_tl",
    ).mse
    for c in cfg.compounds
}

# %%

universal_per_compound_mses = universal_TL_mses()["per_compound"]

# %%

mse_imporvement = {
    c: (universal_per_compound_mses[c] / per_compound_tl_mse[c]) for c in cfg.compounds
}

fig, ax = plt.subplots(figsize=(8, 6))

from src.models.trained_models import MeanModel

ax.scatter(
    # [sum(data_sizes.values()) - data_sizes[c] for c in cfg.compounds],
    [
        (per_compound_tl_mse[c] - universal_per_compound_mses[c])
        / per_compound_tl_mse[c]
        * 100
        for c in cfg.compounds
    ],
    data_sizes.values(),
    alpha=0.5,
)

# # add text with compound names
for c in cfg.compounds:
    ax.text(
        (per_compound_tl_mse[c] - universal_per_compound_mses[c])
        / per_compound_tl_mse[c]
        * 100,
        data_sizes[c],
        c,
    )

ax.set_xlabel("MSE Improvement (Per Compound TL vs Universal FEFF)")
ax.set_xticklabels([f"{i:.0f}%" for i in ax.get_xticks()])
ax.set_ylabel("Data Size")


# %%

vasp_compounds = ["Cu", "Ti"]
vasp_per_compound_tl_mse = {
    c: Trained_FCModel(
        query=DataQuery(compound=c, simulation_type="VASP"),
        name="per_compound_tl",
    ).mse
    for c in vasp_compounds
}
vasp_ft_tl_mse = {
    c: Trained_FCModel(
        query=DataQuery(compound=c, simulation_type="VASP"),
        name="ft_tl",
    ).mse
    for c in vasp_compounds
}
# %%

# grouped bar plot for vasp
fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.35
bar1 = np.arange(len(vasp_compounds))
bar2 = [i + bar_width for i in bar1]
ax.bar(bar1, vasp_per_compound_tl_mse.values(), bar_width, label="Per Compound TL")
ax.bar(bar2, vasp_ft_tl_mse.values(), bar_width, label="FT TL")
ax.set_xticks(bar1 + bar_width / 2)
ax.set_xticklabels(vasp_compounds)
ax.legend()

# %%
