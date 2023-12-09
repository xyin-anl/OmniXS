import os

import numpy as np

from src.data.feff_data import FEFFData
from src.data.raw_data import RAWDataFEFF, RAWDataVASP
from src.data.vasp_data import VASPData

if __name__ == "__main__":
    compound = "Ti"  # <--- change this for different compound

    feff_raw_data = RAWDataFEFF(compound=compound)
    vasp_raw_data = RAWDataVASP(compound=compound)

    feff_ids = set(feff_raw_data.parameters.keys())
    vasp_ids = set(vasp_raw_data.parameters.keys())
    common_ids = feff_ids.intersection(vasp_ids)

    for id in common_ids:
        vasp_data = VASPData(compound, vasp_raw_data.parameters[id])
        feff_data = FEFFData(compound, feff_raw_data.parameters[id])

        # to same energy grid
        vasp_data.resample()
        feff_data.resample()

        # save
        dir_name = f"transfer_learning_data_{compound}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        file_name = "_site_".join(id) + ".txt"
        path = os.path.join(dir_name, file_name)
        np.savetxt(
            path,
            np.vstack((vasp_data.energy, vasp_data.spectra, feff_data.spectra)).T,
            fmt="%s",
            delimiter="\t",
            header="energy\tvasp\tfeff",
        )
