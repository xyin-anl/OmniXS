from pydantic import Field
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


from omegaconf import OmegaConf


class ModelTag(DataTag):
    name: str = Field(..., description="Name of the model")


class TrainedModelLoader:
    file_handler: FileHandler = DEFAULTFILEHANDLER
    hparams = OmegaConf.load("config/training/hparams.yaml").hparams
    input_dim = 64  # TODO: remove hardcoding of i/o dims
    output_dim = 141

    @staticmethod
    def load_model(tag: ModelTag):
        return PlModule.load_from_checkpoint(
            checkpoint_path=TrainedModelLoader.get_ckpt_path(tag),
            model=XASBlock(**TrainedModelLoader.get_layer_widths(tag)),
        )

    @staticmethod
    def get_layer_widths(tag: ModelTag):
        if tag.name != "tunedUniversalXAS":
            hidden_widths = TrainedModelLoader.hparams[tag.name][tag.type][
                tag.element
            ].widths
        else:
            hidden_widths = TrainedModelLoader.hparams["universalXAS"]["FEFF"][
                "All"
            ].widths  # TODO: hacky

        return dict(
            input_dim=TrainedModelLoader.input_dim,
            hidden_dims=hidden_widths,
            output_dim=TrainedModelLoader.output_dim,
        )

    @staticmethod
    def get_ckpt_path(tag: ModelTag):
        paths = TrainedModelLoader.file_handler.serialized_objects_filepaths(
            "TrainedXASBlock", **tag.dict()
        )
        if len(paths) != 1:
            raise ValueError(f"Expected 1 path, got {len(paths)}")
        return paths[0]

    @staticmethod
    def load_ml_splits(tag: ModelTag):
        return TrainedModelLoader.file_handler.deserialize_json(
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
