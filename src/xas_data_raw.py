import numpy as np
from collections import deque


import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Literal, Optional, Dict, Set
import warnings
import logging


@dataclass
class RAWData:
    compound: str
    simulation_type: Literal["VASP", "FEFF"]
    base_dir: Optional[str] = field(default=None, init=False)
    missing_data: Optional[Set[str]] = field(default_factory=set, init=False)

    def __post_init__(self):
        if self.base_dir is None:
            self.base_dir = self._default_base_dir()
        self.parameters  # initialize cached_property

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
    def parameters(self):
        """Parameters for each site in each id where all data is available"""
        parameters = {}
        for id in self._ids:
            E_GS_val = self.E_GS(id)
            if E_GS_val is None:
                sites_with_id = [(id, site) for site in self._sites[id]]
                self.missing_data.update(sites_with_id)
                continue
            for site in self._sites.get(id, []):
                mu_val = self.mu(id, site)
                e_cbm_val = self.e_cbm(id, site)
                E_ch_val = self.E_ch(id, site)
                if mu_val is None or e_cbm_val is None or E_ch_val is None:
                    self.missing_data.add((id, site))
                    continue
                parameters[id, site] = {
                    "mu": mu_val,
                    "e_cbm": e_cbm_val,
                    "E_ch": E_ch_val,
                    "E_GS": E_GS_val,
                    "e_core": self.e_core,
                }
        if len(self.missing_data) > 0:
            warnings.warn(f"{len(self.missing_data)} missing data for {self.compound}")
        return parameters

    @cached_property
    def total_sites(self):
        return len([s for sites in self._sites.values() for s in sites])

    def __len__(self):
        return self.total_sites

    @cached_property
    def _ids(self):
        """Can include ids with missing data"""
        ids = os.listdir(self.base_dir)
        id_filter = re.compile(r"mp-\d+")
        ids = list(filter(lambda x: id_filter.match(x), ids))
        if len(ids) == 0:
            raise ValueError(f"No ids found for {self.compound}")
        return ids

    @cached_property
    def _sites(self):
        """Can include sites with missing data"""
        sites = {}
        for id in self._ids:
            dir_path = os.path.join(self.base_dir, id, self.simulation_type)
            folders = os.listdir(dir_path)
            pattern = r"(\d+)_" + self.compound
            site_list = list(filter(lambda x: re.match(pattern, x), folders))
            if len(site_list) == 0:
                warnings.warn(f"No sites found for {id} at {folders}")
            sites[id] = site_list
        return sites

    @cached_property
    def e_core(self):
        values = {
            "Cu": -8850.2467,
            "Ti": -4864.0371,
        }
        return values[self.compound]

    def mu(self, id, site):
        file_path = os.path.join(
            self.base_dir,
            id,
            self.simulation_type,
            site,
            "mu2.txt",
        )
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            mu = f.readlines()
            mu = list(map(lambda x: x.strip().split(), mu))
            mu = np.array(mu, dtype=float)
            if len(mu) == 0:
                raise ValueError(f"mu2.txt is empty for {file_path}")
            return mu

    def e_cbm(self, id, site):
        file_path = os.path.join(
            self.base_dir,
            id,
            self.simulation_type,
            site,
            "efermi.txt",
        )
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            e_cbm = f.readline()
            if e_cbm == "":
                return None
            return float(e_cbm)

    def E0(self, file_path):
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            last_line = deque(f, maxlen=1)
            if len(last_line) == 0:
                return None
            last_line = last_line.pop()
            reg_pattern = r"E0=\s*(-?\d*\.\d*E[+-]\d*)\s"
            match = re.search(reg_pattern, last_line)
            if not match:
                return None
            return float(match.group(1))

    def E_ch(self, id, site):
        return self.E0(
            os.path.join(
                self.base_dir,
                id,
                self.simulation_type,
                site,
                "OSZICAR",
            )
        )

    def E_GS(self, id):
        return self.E0(
            os.path.join(
                self.base_dir,
                id,
                self.simulation_type,
                "SCF",
                "OSZICAR",
            )
        )


if __name__ == "__main__":
    # data = RAWData("Cu", "VASP")
    # data.parameters

    data = RAWData("Ti", "VASP")
    data.parameters
    print(f"data: {len(data)}, missing: {len(data.missing_data)}")
