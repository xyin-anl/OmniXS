# %%

import numpy as np

from refactor.data.ml_data import MLSplits
from refactor.scripts.plots.constants import FEFFSplits, VASPSplits


def baseline_mse(split: MLSplits):
    prediction = np.array(
        [np.mean(split.train.y, axis=0) for x in range(len(split.test.y))]
    )
    targets = split.test.y
    mse = np.mean((prediction - targets) ** 2)
    return mse


print({k: baseline_mse(v) for k, v in FEFFSplits.items()})
print({k: baseline_mse(v) for k, v in VASPSplits.items()})
