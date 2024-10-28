from omegaconf import OmegaConf
from pydantic import Field

from refactor.data import (DataTag, IdentityScaler, MLSplits, ScaledMlSplit,
                           ThousandScaler)
from refactor.model.training import PlModule
from refactor.model.xasblock import XASBlock
from refactor.utils.io import DEFAULTFILEHANDLER, FileHandler


class ModelTag(DataTag):
    name: str = Field(..., description="Name of the model")


class TrainedModelLoader:

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
