import numpy as np
from utils.src.misc import icecream
from src.data.feff_data_raw import RAWDataFEFF
from src.data.vasp_data_raw import RAWDataVASP


def get_unique_ids_and_counts(ids):
    mat_ids = np.array([x[0] for x in ids])
    unique_ids, id_count = np.unique(mat_ids, return_counts=True)
    return list(zip(unique_ids, id_count))


def partition_ids(unique_ids_and_count, target_sums):
    unique_ids_and_count = sorted(
        unique_ids_and_count, key=lambda x: x[1], reverse=True
    )
    c = len(target_sums)
    groups = [[] for _ in range(c)]
    current_sums = [0] * c
    for unique_id, count in unique_ids_and_count:
        diffs = [abs(current_sums[i] + count - target_sums[i]) for i in range(c)]
        min_group = diffs.index(max(diffs))
        groups[min_group].append(unique_id)
        current_sums[min_group] += count
    return groups


def assign_ids_to_groups(ids, groups):
    grouped_ids = {group: [] for group in range(len(groups))}
    for id in ids:
        # mat_id = id.split("_site_")[0]
        mat_id = id[0]
        for group_index, group in enumerate(groups):
            if mat_id in group:
                grouped_ids[group_index].append(id)
                break
    return grouped_ids[0], grouped_ids[1], grouped_ids[2]


def greedy_multiway_partition(ids, target_sums):
    np.random.shuffle(ids)
    unique_ids_and_count = get_unique_ids_and_counts(ids)
    groups = partition_ids(unique_ids_and_count, target_sums)
    return assign_ids_to_groups(ids, groups)


if __name__ == "__main__":
    compound = "Ti"
    feff_raw_data = RAWDataFEFF(compound=compound)

    ids = feff_raw_data.ids
    target_fractions = [0.8, 0.1, 0.1]
    target_sums = [int(x * len(ids)) for x in target_fractions]

    out = greedy_multiway_partition(ids, target_sums)
    total = sum([len(x) for x in out])

    fractions = [len(x) / total for x in out]
    fractions = [round(x, 2) for x in fractions]

    ic(target_sums)
    ic(total)
    ic(target_fractions)
    ic(fractions)
