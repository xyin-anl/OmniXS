# %%
import numpy as np
from src.data.ml_data import load_xas_ml_data, DataQuery
from config.defaults import cfg
from src.data.spectra_table import SpectraTable
import numpy as np
import matplotlib.pyplot as plt

# %%

import os
import datetime
from pathlib import Path


def get_time_range(vasp_dir):
    start_files = ["INCAR", "KPOINTS", "POSCAR"]
    end_files = ["OSZICAR", "ecorehole.txt", "efermi.txt", "vasp.out"]

    earliest_time = float("inf")
    latest_time = 0

    for file in start_files:
        file_path = vasp_dir / file
        if file_path.exists():
            mod_time = os.path.getmtime(file_path)
            earliest_time = min(earliest_time, mod_time)

    for file in end_files:
        file_path = vasp_dir / file
        if file_path.exists():
            mod_time = os.path.getmtime(file_path)
            latest_time = max(latest_time, mod_time)

    return earliest_time, latest_time


def calculate_runtime(vasp_dir):
    earliest_time, latest_time = get_time_range(vasp_dir)
    if earliest_time == float("inf") or latest_time == 0:
        return None
    return latest_time - earliest_time


def process_material(material_dir):
    vasp_dir = material_dir / "VASP"
    if not vasp_dir.exists():
        return []

    runtimes = []

    for subdir in vasp_dir.iterdir():
        if subdir.is_dir():
            runtime = calculate_runtime(subdir)
            if runtime is not None:
                runtimes.append(runtime)

    return runtimes


# Specify the root directory containing all material folders
root_dir = Path("/Users/bnl/Downloads/dev/aimm/ai_for_xas/dataset/VASP-raw-data/Ti/")

all_runtimes = []

for material_dir in root_dir.iterdir():
    if material_dir.is_dir() and material_dir.name.startswith("mp-"):
        all_runtimes.extend(process_material(material_dir))

# Print all runtimes, one per line
for runtime in all_runtimes:
    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

# Optionally, save results to a file
with open("vasp_runtimes.txt", "w") as f:
    for runtime in all_runtimes:
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}\n")

print(f"\nProcessed {len(all_runtimes)} calculations.")
print("Results have been saved to 'vasp_runtimes.txt'")

# %%

all_runtimes = np.array(all_runtimes)
runtimes = all_runtimes[
    np.logical_and(
        all_runtimes > np.quantile(all_runtimes, 0.2),
        all_runtimes < np.quantile(all_runtimes, 0.8),
    )
]
plt.hist(runtimes, bins=100, density=True)
ax = plt.gca()
ax.set_title(
    f" Est. VASP compute time \n Ti VASP \n Avg time: {np.mean(runtimes):.2f} s\n Count: {len(runtimes)}"
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Density")
