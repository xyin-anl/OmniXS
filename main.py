import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from rich import print


@hydra.main(version_base=None, config_path="cfg", config_name="experiment")
def main(cfg: DictConfig):
    print(cfg)
    data_module = instantiate(cfg.data_module)
    model = instantiate(cfg.pl_module)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
