# %%

from src.models.trained_models import MeanModel
from src.data.ml_data import DataQuery, load_xas_ml_data

# %%

from config.defaults import cfg

medians = {}
for compound in cfg.compounds:
    medians[compound] = MeanModel(DataQuery(compound, "FEFF")).median_of_mse_per_spectra

# %%

medians["Ti VASP"] = MeanModel(DataQuery("Ti", "VASP")).median_of_mse_per_spectra
medians["Cu VASP"] = MeanModel(DataQuery("Cu", "VASP")).median_of_mse_per_spectra

# %%

# medians
# {'Ti': 0.038588285,
#  'V': 0.040166765,
#  'Cr': 0.035693802,
#  'Mn': 0.039994,
#  'Fe': 0.023434678,
#  'Co': 0.019394644,
#  'Ni': 0.02045306,
#  'Cu': 0.010533443,
#  'Ti VASP': 0.15662938,
#  'Cu VASP': 0.023417398}

# save as latex with round to 3 decimals, pad with zero upto 3 decimals
import pandas as pd

df = pd.DataFrame(medians, index=["MSE"])
# df = df.round(4)
df = df.T

# df.to_latex("median_mse.tex", formatters=[lambda x: f"{x:.3f}"]) # IndexError: list index out of range

df.to_latex(
    "median_mse.tex", formatters=[lambda x: f"{x:.4f}" for _ in range(len(df.columns))]
)
