from tqdm import tqdm

from src.data.feff_data import FEFFData
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data import VASPData
from src.data.vasp_data_raw import RAWDataVASP

if __name__ == "__main__":
    compound = "Ti"
    simulation_type = "FEFF"
    raw_data_class = RAWDataFEFF if simulation_type == "FEFF" else RAWDataVASP
    raw_data = raw_data_class(compound=compound)
    ids = raw_data.parameters.keys()
    ids = tqdm(ids)  # progress bar
    for id in ids:
        data_class = FEFFData if simulation_type == "FEFF" else VASPData
        data = data_class(
            compound=compound,
            params=raw_data.parameters[id],
            id=id,
        )
        data.save()
        # plt.plot(data.energy, data.spectra, label=id)
        # plt.legend()
    # plt.show()
