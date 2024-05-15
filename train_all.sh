#!/bin/bash

nohup_run() {
    nohup ./train.sh "$1" "$2" "$3" >"$1_$2_$3.out" 2>&1 &
}

./train.sh "ALL" "FEFF" "universal_tl"

feffcompounds=("Co" "Cr" "Cu" "Fe" "Mn" "Ni" "Ti" "V")
vaspcompounds=("Cu" "Ti")
model_names=("per_compound_tl" "ft_tl")
for model_name in "${model_names[@]}"; do
    for compound in "${feffcompounds[@]}"; do
        nohup_run "$compound" "FEFF" "$model_name"
    done
    for compound in "${vaspcompounds[@]}"; do
        nohup_run "$compound" "VASP" "$model_name"
    done
done
