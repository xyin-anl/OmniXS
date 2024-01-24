import sys
import logging
import optuna
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from utils.src.lightning.pl_module import PLModule

from rich import print
import torch


class FC_XAS(nn.Module):
    def __init__(self, widths=[64, 100, 141], dropout_rate=0.5):
        super().__init__()
        self.widths = widths
        self.pairs = [(w1, w2) for w1, w2 in zip(self.widths[:-1], self.widths[1:])]
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.relu = nn.ReLU()
        for i, (w1, w2) in enumerate(self.pairs):
            self.layers.append(nn.Linear(w1, w2))
            if i != len(self.pairs) - 1:
                self.batch_norms.append(nn.BatchNorm1d(w2))
                self.dropouts.append(nn.Dropout(p=dropout_rate))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.pairs) - 1:
                # x = self.batch_norms[i](x)
                x = x * torch.sigmoid(x)  # swish activation
                # x = self.dropouts[i](x)
        return x

    # main()


from src.data.ml_data import XASPlData, DataQuery
from lightning import Trainer
import lightning
import lightning as pl


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    # model = instantiate(cfg.model)
    model = PLModule(FC_XAS(widths=[64, 100, 141]))
    data_module = instantiate(cfg.data_module)

    # TODO: Fix this error: it is sending removing the batch dimension for some reason
    # model.example_input_array = data_module.train_dataloader().dataset[0][0].to("mps")

    trainer = instantiate(cfg.trainer)

    # TODO: Fix this error: this is getting trainer initilization stuck
    # trainer.callbacks = instantiate(cfg.callbacks).values()

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)


class Optimizer:
    def __init__(self, hydra_configs):
        self.cfg = hydra_configs

    def optimize(self, trial):
        # depth = trial.suggest_int("depth", 1, 3)
        depth = trial.suggest_int(
            "depth",
            self.cfg.optuna.params.min_depth,
            self.cfg.optuna.params.max_depth,
        )

        widths = [
            trial.suggest_int(
                f"width_{i}",
                self.cfg.optuna.params.min_width,
                self.cfg.optuna.params.max_width,
                self.cfg.optuna.params.step_width,
            )
            for i in range(depth)
        ]

        # widths = [64] + widths
        # self.cfg["model"]["model"]["widths"] = widths
        # learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        # self.cfg["model"]["learning_rate"] = learning_rate
        # model = instantiate(self.cfg.model)

        # batch_exponent = trial.suggest_int("batch", 6, 9, 1)
        # batch_size = 2**batch_exponent
        # self.cfg["data_module"]["batch_size"] = batch_size

        widths = [64] + widths + [141]
        model = PLModule(FC_XAS(widths=widths))

        data_module = instantiate(self.cfg.data_module)

        # TODO: Fix this error: it is sending removing the batch dimension for some reason
        # model.example_input_array = (
        #     data_module.train_dataloader().dataset[0][0].to("mps")
        # )  # needed for loggers

        # traint and test
        trainer = instantiate(self.cfg.trainer)

        # TODO: Fix this error: this is getting trainer initilization stuck
        # trainer.callbacks = []  # removes progressbar
        # for cb_name, cb_cfg in self.cfg.callbacks.items():
        #     cb_instance = instantiate(cb_cfg)
        #     trainer.callbacks.append(cb_instance)

        # TODO: make this load from cfg after fixing the error above
        trainer.callbacks.extend(
            [
                lightning.pytorch.callbacks.lr_finder.LearningRateFinder(),
                lightning.pytorch.callbacks.early_stopping.EarlyStopping(
                    monitor="val_loss", patience=10, mode="min", verbose=False
                ),
            ]
        )

        # trainer.callbacks.append(
        #     PyTorchLightningPruningCallback(trial, monitor="val_loss")
        # )

        trainer.fit(model, data_module)

        return trainer.callback_metrics["val_loss"]


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def run_optmization(cfg: DictConfig):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=cfg.optuna.storage,
        # storage="sqlite:///{}.db".format("test"),
        load_if_exists=True,
    )
    optimizer_class = Optimizer(hydra_configs=cfg)
    study.optimize(optimizer_class.optimize, n_trials=cfg.optuna.n_trials)
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best Trial: ", study.best_trial)
    # print("Trials: ", study.trials)


if __name__ == "__main__":
    run_optmization()
    # main()
