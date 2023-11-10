import os
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Optional

import numpy as np
from scipy.stats import cauchy

from src.raw_data import RAWData


@dataclass
class RAWDataFEFF(RAWData):
    compound: str
    base_dir: Optional[str] = field(default=None, init=False)
    simulation_type: Literal["FEFF"] = field(default="FEFF", init=True)
    intermediate_dir: Literal["FEFF-XANES"] = field(default="FEFF-XANES", init=True)

    def __post_init__(self):
        if self.base_dir is None:
            self.base_dir = self._default_base_dir()
        self.parameters  # initialize cached property

    def mu(self, id, site):
        file_path = os.path.join(
            self.base_dir,
            id,
            f"{self.simulation_type}-XANES",
            site,
            "xmu.dat",
        )
        if not os.path.exists(file_path):
            warnings.warn(f"Missing mu for {id} at {site}")
            return None
        with open(file_path, "r") as f:
            lines = f.readlines()
            line = list(filter(lambda x: x.startswith("#  xsedge+"), lines))[0]
            normalization = float(line.split()[-1])
            data = np.loadtxt(file_path)
            energy = data[:, 0]
            mu = data[:, 3] * normalization
            if mu is None:
                warnings.warn(f"Missing mu for {id} at {site}")
                return None
            return np.array([energy, mu]).T

    @cached_property
    def parameters(self):
        parameters = {}
        for id in self._ids:
            for site in self._sites.get(id, []):
                mu_val = self.mu(id, site)
                if mu_val is None:
                    self.missing_data.add((id, site))
                    continue
                parameters[id, site] = {"mu": mu_val}
        if len(self.missing_data) > 0:
            warnings.warn(f"{len(self.missing_data)} missing data for {self.compound}")
        return parameters


if __name__ == "__main__":
    data = RAWDataFEFF(compound="Ti")
    data.parameters
    print(f"data: {len(data)}, missing: {len(data.missing_data)}")

    id = next(iter(data._ids))
    site = next(iter(data._sites[id]))
    id = ("mp-390", "000_Ti")  # reference to another paper data

    from matplotlib import pyplot as plt

    plt.plot(
        data.parameters[id]["mu"][:, 0],
        data.parameters[id]["mu"][:, 1],
    )
    plt.show()

# %%
