# %%
from omegaconf import OmegaConf
from pydantic import Field
from torch import nn

from refactor.data import (
    DataTag,
    IdentityScaler,
    MLSplits,
    ScaledMlSplit,
    ThousandScaler,
)
from refactor.model.training import PlModule
from refactor.model.xasblock import XASBlock
from refactor.utils.io import DEFAULTFILEHANDLER, FileHandler


class ModelTag(DataTag):
    name: str = Field(..., description="Name of the model")


class TrainedModelLoader:

    @staticmethod
    def load_model_for_finetuning(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
        freeze_first_k_layers: int = 0,
    ) -> PlModule:
        model = TrainedModelLoader.load_model(tag, file_handler)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        layer_count = 0
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                if layer_count < freeze_first_k_layers:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_running_stats()

                if isinstance(m, nn.Linear):
                    layer_count += 1

        return model

    @staticmethod
    def load_model(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ) -> PlModule:
        return PlModule.load_from_checkpoint(
            checkpoint_path=TrainedModelLoader.get_ckpt_path(tag, file_handler),
            model=XASBlock(**TrainedModelLoader.get_layer_widths(tag)),
        )

    @staticmethod
    def get_layer_widths(
        tag: ModelTag,
        hparams: dict = OmegaConf.load("config/training/hparams.yaml").hparams,
        input_dim: int = 64,
        output_dim: int = 141,
    ):
        if tag.name != "tunedUniversalXAS":
            hidden_widths = hparams[tag.name][tag.type][tag.element].widths
        else:
            hidden_widths = hparams["universalXAS"]["FEFF"]["All"].widths  # TODO: hacky

        return dict(
            input_dim=input_dim,
            hidden_dims=hidden_widths,
            output_dim=output_dim,
        )

    @staticmethod
    def get_ckpt_path(
        tag: ModelTag,
        file_handler=DEFAULTFILEHANDLER(),
    ):
        paths = file_handler.serialized_objects_filepaths(
            "TrainedXASBlock", **tag.dict()
        )
        if len(paths) != 1:
            raise ValueError(f"Expected 1 path, got {len(paths)}")
        return paths[0]

    @staticmethod
    def load_ml_splits(
        tag: ModelTag,
        file_handler: FileHandler = DEFAULTFILEHANDLER(),
    ):
        return file_handler.deserialize_json(
            MLSplits,
            DataTag(element=tag.element, type=tag.type),
        )

    @staticmethod
    def load_scaled_splits(
        tag: ModelTag, x_scaler: type = IdentityScaler, y_scaler: type = ThousandScaler
    ):
        splits = TrainedModelLoader.load_ml_splits(tag)
        return ScaledMlSplit(
            x_scaler=x_scaler(),
            y_scaler=y_scaler(),
            **splits.dict(),
        )


# %%
if __name__ == "__main__":

    def param_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def linear_param_count(model):
        count = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                count += m.weight.numel()
                count += m.bias.numel()
        return count

    for count in range(4):
        module = TrainedModelLoader.load_model_for_finetuning(
            tag=ModelTag(
                element="All",
                type="FEFF",
                name="universalXAS",
            ),
            freeze_first_k_layers=count,
        )
        param_fn = param_count
        trainable_params = param_fn(module)
        print(f"Trainable parameters with {count} frozen layers: {trainable_params}")

        print(
            [
                layer.__class__.__name__
                for layer in module.modules()
                if hasattr(layer, "weight") and layer.weight.requires_grad
            ]
        )
        print("=====================================")

    # %%

    from refactor.data import FEFFDataTags

    min = 1e10
    min_elem = None
    for data_tag in FEFFDataTags:
        model_tag = ModelTag(name="expertXAS", **data_tag.dict())
        xasblock = XASBlock(**TrainedModelLoader.get_layer_widths(model_tag))
        param_fn = param_count
        if param_fn(xasblock) < min:
            min_elem = data_tag
            min = param_fn(xasblock)
        print(data_tag, param_fn(xasblock))
    print(min_elem, min)

    # %%
