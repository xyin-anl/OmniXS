# %%

%load_ext autoreload
%autoreload 2


from matplotlib import pyplot as plt

from src.analysis.plots import Plot
from src.data.ml_data import DataQuery
from src.models.trained_models import LinReg, Trained_FCModel
from utils.src.misc.icecream import ic
# %%


query = DataQuery("Cu", "FEFF")
fc_model = Trained_FCModel(query)
lin_model = LinReg(query)
Plot().bar_plot_of_loss([fc_model])
Plot().bar_plot_of_loss([lin_model])

# %%
