import numpy as np
import yaml
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, Set


@dataclass
class RAWData(ABC):
    compound: str = field(default=None, init=False)
    simulation_type: str = field(default=None, init=False)
    missing_data: Optional[Set[str]] = field(default_factory=set, init=False)
    intermediate_dir: str = field(default=None, init=False)

    def _default_base_dir(self):
        default_base_dir = os.path.join(  # default
            "dataset",
            f"{self.simulation_type}-raw-data",
            f"{self.compound}",
        )
        if not os.path.exists(default_base_dir):
            raise ValueError(f"{default_base_dir} does not exist")
        return default_base_dir

    @cached_property
    def _ids(self):
        """Can include ids with missing data"""
        ids = os.listdir(self.base_dir)
        id_filter = re.compile(r"(mp-\d+|mvc-\d+)")
        ids = list(filter(lambda x: id_filter.match(x), ids))
        if len(ids) == 0:
            raise ValueError(f"No ids found for {self.compound}")
        return ids

    @property
    def ids(self):
        # includes valid id and site combinations
        # return np.array(["_site_".join(x) for x in list(self.parameters.keys())])
        return list(self.parameters.keys())

    @cached_property
    def _sites(self):
        """Can include sites with missing data"""
        sites = {}
        for id in self._ids:
            dir_path = os.path.join(self.base_dir, id, self.intermediate_dir)
            folders = os.listdir(dir_path)
            pattern = r"(\d+)_" + self.compound
            site_list = list(filter(lambda x: re.match(pattern, x), folders))
            if len(site_list) == 0:
                warnings.warn(f"No sites found for {id} at {folders}")
            sites[id] = site_list
        return sites

    @cached_property
    def total_sites(self):
        return len([s for sites in self._sites.values() for s in sites])

    @staticmethod
    def configs(cfg_path="cfg/transformations.yaml"):
        with open(cfg_path) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        return cfg

    def __len__(self):
        return self.total_sites

    @abstractmethod
    def mu(self, id, site):
        pass

    @abstractmethod
    def parameters(self):
        pass
