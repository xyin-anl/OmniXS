# nohup python main.py compound_name=Cu-O simulation_type=feff >cu-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=feff >ti-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=vasp >ti-o-vasp.log &

# nohup python main.py compound_name=Cu-O simulation_type=feff model.model.widths=[64,180,200] trainer.max_epochs=500 model.learning_rate=0.000815876 >cu-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=feff model.model.widths=[64,190,180] trainer.max_epochs=500 model.learning_rate=0.001069577 >ti-o-feff.log &
# nohup python main.py compound_name=Ti-O simulation_type=vasp model.model.widths=[64,150,120,170] trainer.max_epochs=500 model.learning_rate=0.00436152122 >ti-o-vasp.log &

python main.py compound_name=Ti-O simulation_type=vasp model.model.widths=[64,150,120,170] trainer.max_epochs=500 model.learning_rate=0.00436152122 callbacks.early_stopping.patience=100 >ti-o-vasp.log