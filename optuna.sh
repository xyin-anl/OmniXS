# nohup python main.py compound_name=Cu-O simulation_type=feff >cu-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=feff >ti-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=vasp >ti-o-vasp.log &

# nohup python main.py compound_name=Cu-O simulation_type=feff model.model.widths=[64,180,200] trainer.max_epochs=500 model.learning_rate=0.000815876 >cu-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=feff model.model.widths=[64,190,180] trainer.max_epochs=500 model.learning_rate=0.001069577 >ti-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=vasp model.model.widths=[64,150,120,170] trainer.max_epochs=500 model.learning_rate=0.00436152122 >ti-o-vasp.log &

# python main.py compound_name=Ti-O simulation_type=vasp model.model.widths=[64,150,120,170] trainer.max_epochs=500 model.learning_rate=0.00436152122 callbacks.early_stopping.patience=100 >ti-o-vasp.log

# python main.py compound_name=Ti-O simulation_type=feff data_module.query.split=random trainer.max_epochs=500 callbacks.early_stopping.patience=3 >ti-o-vasp_random_split.log

# python main.py compound_name=Cu simulation_type=FEFF trainer.max_epochs=500 callbacks.early_stopping.patience=3 >Cu_FEFF.log &

compounds=("Co" "Cr" "Cu" "Fe" "Mn" "Ni" "Ti" "V")
for compound in "${compounds[@]}"; do
    nohup python main.py compound_name=$compound simulation_type=FEFF trainer.max_epochs=500 callbacks.early_stopping.patience=3 >$compound"_FEFF.log" &
done

# optimal_fc_params:
#   Cu-O:
#     FEFF: [64, 180, 200]
#   Ti-O:
#     FEFF: [64, 190, 180]
#     VASP: [64, 150, 120, 170]
