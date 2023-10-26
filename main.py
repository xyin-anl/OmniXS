from optuna.integration import PyTorchLightningPruningCallback
import sys
import logging
import optuna
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rich import print


@hydra.main(version_base=None, config_path="cfg", config_name="experiment")
def main(cfg: DictConfig):
    # print(cfg)
    # return
    data_module = instantiate(cfg.data_module)
    model = instantiate(cfg.model)

    model.example_input_array = data_module.train_dataloader().dataset[0][0].to("mps")

    # from utils.src.misc.model_adapters import PLAdapter
    # print(PLAdapter(model).summary())

    trainer = instantiate(cfg.trainer)
    trainer.callbacks = instantiate(cfg.callbacks).values()
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)


class Optimizer:
    def __init__(self, hydra_configs):
        self.cfg = hydra_configs

    def optimize(self, trial):
        depth = trial.suggest_categorical("depth", [1, 2, 3])
        widths = [
            trial.suggest_int(
                f"width_{i}",
                self.cfg.optuna.params.min_width,
                self.cfg.optuna.params.max_width,
                self.cfg.optuna.params.step_width,
            )
            for i in range(depth)
        ]
        widths = [64] + widths
        self.cfg["model"]["model"]["widths"] = widths
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        self.cfg["model"]["learning_rate"] = learning_rate
        model = instantiate(self.cfg.model)

        # data
        self.cfg["data_module"]["batch_size"] = trial.suggest_categorical(
            "batch_size", [128]
        )
        data_module = instantiate(self.cfg.data_module)
        model.example_input_array = (
            data_module.train_dataloader().dataset[0][0].to("mps")
        )  # needed for loggers

        # traint and test
        self.cfg["logger"]["tensorboard"]["save_dir"] = (
            self.cfg["logger"]["tensorboard"]["save_dir"] + f"/{trial.number}"
        )
        trainer = instantiate(self.cfg.trainer)

        for cb_name, cb_cfg in self.cfg.callbacks.items():
            cb_instance = instantiate(cb_cfg)
            trainer.callbacks.append(cb_instance)

        trainer.callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="val_loss")
        )

        trainer.fit(model, data_module)

        return trainer.callback_metrics["val_loss"]


@hydra.main(version_base=None, config_path="cfg", config_name="experiment")
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
    print("Trials: ", study.trials)


if __name__ == "__main__":
    run_optmization()
