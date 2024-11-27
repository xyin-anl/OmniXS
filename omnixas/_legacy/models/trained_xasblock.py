from config.defaults import cfg
from _legacy.models.trained_model import TrainedModel
from omnixas.model.xasblock import XASBlock


import optuna
import torch


import os
from functools import cached_property

from _legacy.data.ml_data import DataQuery


class TrainedXASBlock(TrainedModel):
    def __init__(self, query, name, date_time=None, version=None, ckpt_name=None):
        super().__init__(query)
        self._name = name
        self.date_time = date_time or self._latest_dir(self._hydra_dir)
        self.version = (
            version or self._latest_dir(self._lightning_log_dir).split("_")[-1]
        )  # TODO: make it try optuna study
        self.ckpt_name = ckpt_name

    @property
    def name(self):
        return self._name

    @cached_property
    def model(self):
        # model = instantiate(cfg.model)

        model_params = torch.load(self._ckpt_path)
        # change keys of state_dict to remove the "model." prefix
        model_params["state_dict"] = {
            k.replace("model.", ""): v for k, v in model_params["state_dict"].items()
        }

        # infer widths from state_dict
        # TODO: double check this
        state = model_params["state_dict"]
        weight_shapes = [
            state[k].shape for k in state.keys() if "layers" in k and "weight" in k
        ]

        input_sz = weight_shapes[0][1]
        hidden_sz = [w[0] for w in weight_shapes[:]]
        widths = [input_sz] + hidden_sz  # output is already included

        model = XASBlock(widths=widths, input_dims=None, output_dim=None)

        # model = FC_XAS(
        #     widths=hidden_sz,
        #     input_dim=None,
        #     output_dim=None,
        #     compound=self.compound,
        #     simulation_type=self.simulation_type,
        # )

        model.load_state_dict(model_params["state_dict"])
        model.eval()
        return model

    @cached_property
    def optuna_study(self):
        kwargs = dict(compound=self.compound, simulation_type=self.simulation_type)
        # TODO: move this config to yaml
        study_name = f"{self.compound}-{self.simulation_type}"
        storage = cfg.paths.optuna_db.format(**kwargs)
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study

    @cached_property
    def predictions(self):
        return self.model(torch.Tensor(self.data.test.X)).detach().numpy()

    def _latest_dir(self, directory):
        assert os.path.exists(directory), f"Directory {directory} does not exist"
        all_items = os.listdir(directory)
        dirs = [  # Filter out items that are not directories
            item for item in all_items if os.path.isdir(os.path.join(directory, item))
        ]
        assert len(dirs) > 0, f"Directory {directory} is empty"
        dirs.sort(  # Sort directories by creation time
            key=lambda x: os.path.getctime(os.path.join(directory, x)),
            reverse=True,
        )
        return dirs[0]

    @property
    def _hydra_dir(self):
        hydra_dir = "logs/{compound}_{simulation_type}/runs/".format(
            **self.query.__dict__
        )
        hydra_dir = hydra_dir.replace("runs", self.name)  # adhoc TODO:
        assert os.path.exists(hydra_dir), f"Hydra dir {hydra_dir} not found"
        return hydra_dir

    @property
    def _lightning_log_dir(self):
        lightning_dir = self._hydra_dir + self.date_time + "/lightning_logs/"
        assert os.path.exists(lightning_dir), f"lightning_dir {lightning_dir} not found"
        return lightning_dir

    @cached_property
    def _ckpt_path(self, version=None):
        log_dir = self._lightning_log_dir
        version_dir = f"version_{self.version}"
        # if ckpt path is none select one that starts with "epoch*.ckpt"

        # ckpt_path = log_dir + version_dir + f"/checkpoints/{self.ckpt_name}.ckpt"
        ckpt_path = log_dir + version_dir + "/checkpoints/"
        if self.ckpt_name is None:
            ckpt_path += [f for f in os.listdir(ckpt_path) if f.startswith("epoch")][0]
        else:
            ckpt_path += self.ckpt_name + ".ckpt"
        assert os.path.exists(ckpt_path), f"ckpt_path {ckpt_path} not found"
        return ckpt_path


class PreTrainedFCXASModel(XASBlock):
    # Thin Wrapper around Trained_FCModel.model to allow for fine-tuning
    def __init__(
        self,
        query: DataQuery,
        name,
        **model_kwargs,
    ):
        trained_model = TrainedXASBlock(query=query, name=name).model

        super().__init__(**model_kwargs)
        self.load_state_dict(trained_model.state_dict())
