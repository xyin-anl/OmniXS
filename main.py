import sys
import logging
import optuna
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rich import print


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    # print(cfg)
    # return
    data_module = instantiate(cfg.data_module)
    model = instantiate(cfg.model)

    model.example_input_array = data_module.train_dataloader().dataset[0][0].to("mps")

    trainer = instantiate(cfg.trainer)
    trainer.callbacks = instantiate(cfg.callbacks).values()
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)


class Optimizer:
    def __init__(self, hydra_configs):
        self.cfg = hydra_configs

    def optimize(self, trial):
        depth = trial.suggest_int("depth", 1, 3)
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
        self.cfg["model"]["learning_rate"] = learning_rate
        model = instantiate(self.cfg.model)

        # batch_exponent = trial.suggest_int("batch", 6, 9, 1)
        # batch_size = 2**batch_exponent
        # self.cfg["data_module"]["batch_size"] = batch_size

        data_module = instantiate(self.cfg.data_module)
        model.example_input_array = (
            data_module.train_dataloader().dataset[0][0].to("mps")
        )  # needed for loggers

        # traint and test
        trainer = instantiate(self.cfg.trainer)

        trainer.callbacks = []  # removes progressbar
        for cb_name, cb_cfg in self.cfg.callbacks.items():
            cb_instance = instantiate(cb_cfg)
            trainer.callbacks.append(cb_instance)

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
    # run_optmization()
    main()
