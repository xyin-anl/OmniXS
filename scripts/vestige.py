""" 
# =============================================================================
# COMPARE PER SPECTRA ALIGNMENT VS DTW BASED ALIGNMENT
# =============================================================================
compound = "Ti"
raw_data = RAWDataFEFF(compound=compound)
vasp_raw_data = RAWDataVASP(compound=compound)


# seed = 42
# random.seed(seed)
sample_length = 10
ids = set(raw_data.parameters.keys())
vasp_ids = set(vasp_raw_data.parameters.keys())
common_ids = ids.intersection(vasp_ids)

plt.style.use(["default", "science"])
ids = random.choices(list(common_ids), k=sample_length)
fig, axs = plt.subplots(len(ids), 1, figsize=(6, 3 * len(ids)))
time_corr = []
time_dtw = []
for simulation_type in ["VASP", "FEFF"]:
    raw_data = vasp_raw_data if simulation_type == "VASP" else raw_data
    data_class = VASPData if simulation_type == "VASP" else FEFFData
    for ax, id in zip(axs, ids):
        data = data_class(
            compound=compound,
            params=raw_data.parameters[id],
        )
        # data.transform()

        if data.simulation_type == "FEFF":
            t1 = time.time()
            data.align(VASPData(compound, vasp_raw_data.parameters[id]))
            del_t = time.time() - t1
            time_corr.append(del_t)
            ic(del_t)
        ax.plot(
            data.energy,
            data.spectra,
            label=f"{data.simulation_type}_{id}",
            linestyle="-",
        )

        # doing again for dtw based
        if data.simulation_type == "FEFF":
            data = data_class(
                compound=compound,
                params=raw_data.parameters[id],
            )
            t1 = time.time()
            shift = data_class.dtw_shift(
                data,
                VASPData(compound, vasp_raw_data.parameters[id]),
            )
            del_t = time.time() - t1
            time_dtw.append(del_t)
            ic(del_t)
            data.align_energy(-shift)
            # data.truncate_emperically()
            ax.plot(
                data.energy,
                data.spectra,
                label=f"{data.simulation_type}_{id}_dtw",
                linestyle="--",
            )

        ax.legend()
        ax.sharex(axs[0])

axs[-1].set_xlabel("Energy (eV)", fontsize=18)

plt.suptitle(f"Per-spectra alignment samples: {compound}", fontsize=18)
plt.tight_layout()
# plt.savefig(f"vasp_truncation_examples_{compound}.pdf", bbox_inches="tight", dpi=300)
plt.show()
# =============================================================================
"""
