from src.data.pl_data import XASData
from utils.src.optuna.dynamic_fc import PlDynamicFC

# import pytorch_lightning as pl
import lightning as pl
import torch
import yaml


if __name__ == "__main__":
    from utils.src.lightning.loggers.tb.log_train_val_loss import (
        TensorboardLogTestTrainLoss,
    )

    query = {
        "compound": "Ti-O",
        "simulation_type": "VASP",
        "split": "material",
    }
    optimal_fc = yaml.safe_load(open("cfg/misc.yaml", "r"))["optimal_fc_params"]
    widths = optimal_fc[query["compound"]][query["simulation_type"]]
    data_module = XASData(query=query, num_workers=0, batch_size=128)
    model = PlDynamicFC(widths=widths, output_size=200)
    trainer = pl.Trainer(max_epochs=500)
    callbacks = [
        pl.pytorch.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", verbose=False
        ),
        TensorboardLogTestTrainLoss(),
    ]
    trainer.callbacks = trainer.callbacks + callbacks
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)
