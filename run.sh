max_epochs=1

run() {
    python main.py compound_name=$1 simulation_type=$2 trainer.max_epochs=$max_epochs model_name=$3
}

nohup_run() {
    nohup run $1 $2 $3 >$1_$2_$3.out 2>&1 &
}

run "ALL" "FEFF" "universal_tl"

feffcompounds=("Co" "Cr" "Cu" "Fe" "Mn" "Ni" "Ti" "V")
vaspcompounds=("Cu" "Ti")
model_names=("per_compound_tl" "ft_tl")
for model_name in "${model_names[@]}"; do
    for compound in "${feffcompounds[@]}"; do
        nohup_run $compound "FEFF" $model_name
    done
    for compound in "${vaspcompounds[@]}"; do
        nohup_run $compound "VASP" $model_name
    done
done
