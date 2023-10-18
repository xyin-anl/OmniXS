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


if __name__ == "__main__":
    main()
