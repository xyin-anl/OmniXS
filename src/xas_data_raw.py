import numpy as np
from collections import deque


import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Literal, Optional, Dict, Set
import warnings


@dataclass
class RAWData:
    compound: str
    simulation_type: Literal["VASP", "FEFF"]
    base_dir: Optional[str] = field(default=None, init=False)
    e_core: Optional[float] = field(default=None, init=False)
    missing_data: Optional[Set] = field(default_factory=set, init=False, repr=False)
    ids: Optional[Set] = field(default_factory=set, init=False, repr=False)
    site: Optional[Set] = field(default_factory=set, init=False, repr=False)
    mu: Optional[Dict] = field(default_factory=dict, init=False, repr=False)
    e_cbm: Optional[Dict] = field(default_factory=dict, init=False, repr=False)
    E_ch: Optional[Dict] = field(default_factory=dict, init=False, repr=False)
    E_GS: Optional[Dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.base_dir is None:
            default_path = os.path.join(
                "dataset",
                f"{self.simulation_type}-raw-data",
                f"{self.compound}",
            )
            self.base_dir = default_path
        values = {
            "Cu": -8850.2467,
            "Ti": -4864.0371,
        }
        self.e_core = values[self.compound]
        
        # TODO

        if len(self.missing_data) > 0:
            warnings.warn(
                f"Missing {len(self.missing_data)} required files in {self.base_dir}"
            )

    def _ids(self):
        ids = os.listdir(self.base_dir)
        id_filter = re.compile(r"mp-\d+")
        ids = list(filter(lambda x: id_filter.match(x), ids))
        if len(ids) == 0:
            raise ValueError(f"No ids found for {self.compound}")
        return ids

    def _sites(self):
        sites = {}
        for id in self._ids:
            dir_path = os.path.join(self.base_dir, id, self.simulation_type)
            folders = os.listdir(dir_path)
            pattern = r"(\d+)_" + self.compound
            site_list = list(filter(lambda x: re.match(pattern, x), folders))
            if len(site_list) == 0:
                raise ValueError(f"No sites found for {id} at {folders}")
            sites[id] = site_list
        return sites

    def _mu(self):
        mu = {}
        for id in self._ids:
            for site in self._sites[id]:
                mu_val = self._mu_site(id, site)
                if mu_val is None:
                    self.missing_data.add((id, site))
                else:
                    mu[id, site] = self._mu_site(id, site)
        return mu

    def _e_cbm(self):
        e_cbm = {}
        for id in self._ids:
            for site in self._sites[id]:
                e_cbm_val = self._e_cbm_site(id, site)
                if e_cbm_val is None:
                    self.missing_data.add((id, site))
                else:
                    e_cbm[id, site] = self._e_cbm_site(id, site)
        return e_cbm

    def _E_ch(self):
        E_ch = {}
        for id in self._ids:
            for site in self._sites[id]:
                E_ch_val = self._E_ch_site(site, id)
                if E_ch_val is None:
                    self.missing_data.add((id, site))
                else:
                    E_ch[id, site] = self._E_ch_site(site, id)
        return E_ch

    def _E_GS(self):
        E_GS = {}
        for id in self._ids:
            E_GS_val = self._E_GS_site(id)
            if E_GS_val is None:
                update_pairs = [(id, site) for site in self._sites[id]]
                self.missing_data.update(update_pairs)
            else:
                E_GS[id] = self._E_GS_site(id)
        return E_GS

    def _mu_site(self, id, site):
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

    def _e_cbm_site(self, id, site):
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

    def _E0(self, dir_path, file_name):
        file_path = os.path.join(
            dir_path,
            file_name,
        )
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

    def _E_ch_site(self, site, id):
        return self._E0(
            os.path.join(
                self.base_dir,
                id,
                self.simulation_type,
                site,
            ),
            "OSZICAR",
        )

    def _E_GS_site(self, id):
        return self._E0(
            os.path.join(
                self.base_dir,
                id,
                self.simulation_type,
                "SCF",
            ),
            "OSZICAR",
        )


if __name__ == "__main__":
    data = RAWData("Cu", "VASP")
    # print(data)
    # print(data.ids)
    # print(data.sites)
    # print(data.mu)
    # print(data.e_cbm)
    # print(data.E_ch)
    # print(data.E_GS)

    print("dummy")
