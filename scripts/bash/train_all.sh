#!/bin/bash

nohup_run() {
    nohup ./train.sh "$1" "$2" "$3" >"$1_$2_$3.out" 2>&1 &
}



# FIRST RUN
# patiece 50 epoch 500
# ./train.sh "ALL" "FEFF" "universal_tl"
# feffcompounds=("Co" "Cr" "Cu" "Fe" "Mn" "Ni" "Ti" "V")
# vaspcompounds=("Cu" "Ti")
# model_names=("per_compound_tl" "ft_tl")
# for model_name in "${model_names[@]}"; do
#     for compound in "${feffcompounds[@]}"; do
#         nohup_run "$compound" "FEFF" "$model_name"
#     done
#     for compound in "${vaspcompounds[@]}"; do
#         nohup_run "$compound" "VASP" "$model_name"
#     done
# done
#
#

# # SECOND RUN
# # patience 5 epoch 1000
# nohup_run "ALL" "FEFF" "universal_tl"
# nohup_run "Cu" "FEFF" "per_compound_tl"
# nohup_run "Cr" "FEFF" "per_compound_tl"
# nohup_run "Mn" "FEFF" "per_compound_tl"

# THIRD RUN
# patience 5 epoch 1000
feffcompounds=("Co" "Cr" "Cu" "Fe" "Mn" "Ni" "Ti" "V")
for compound in "${feffcompounds[@]}"; do
    nohup_run "$compound" "FEFF" "ft_tl"
done

vaspcompounds=("Cu" "Ti")
for compound in "${vaspcompounds[@]}"; do
    nohup_run "$compound" "VASP" "ft_tl"
    nohup run "$compound" "VASP" "per_compound_tl"
done






