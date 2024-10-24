# %%
from pydantic import BaseModel, Field, field_serializer
from typing import Any, ClassVar, Dict, List, Optional

# skip splits serilziation
from pydantic import model_serializer


import numpy as np
from pydantic import Field, field_validator

from refactor.data.constants import FEFFDataTags, VASPDataTags
from refactor.data.ml_data import DataTag, MLSplits

# from refactor.data.constants import FEFFDataTags, VASPDataTags
from refactor.utils import DEFAULTFILEHANDLER
from refactor.utils.io import DEFAULTFILEHANDLER, FileHandler


class MergedSplits(MLSplits):
    splits: Dict[DataTag, MLSplits] = Field(default_factory=dict)

    @classmethod
    def load(cls, tags: List[DataTag], file_handler: "FileHandler") -> "MergedSplits":
        merged = cls()
        for tag in tags:
            split = file_handler.deserialize_json(MLSplits, supplemental_info=tag)
            merged.append(tag=tag, split=split)
        return merged

    def append(self, tag: DataTag, split: MLSplits):
        for attr in MLSplits.__fields__.keys():
            new_data = getattr(split, attr)
            if new_data is None:
                continue
            existing_data = getattr(self, attr)
            if existing_data is None:
                setattr(self, attr, new_data)
            else:
                existing_data.X = np.concatenate([existing_data.X, new_data.X])
                existing_data.y = np.concatenate([existing_data.y, new_data.y])
            self.splits[tag] = split


class FEFFSplits(MergedSplits):
    def __new__(cls):
        # useful for hydra config
        merged = MergedSplits.load(FEFFDataTags, DEFAULTFILEHANDLER)
        ml_split = MLSplits(
            train=merged.train,
            val=merged.val,
            test=merged.test,
        )
        return ml_split


class VASPSplits(MergedSplits):
    def __new__(cls):
        # useful for hydra config
        return MergedSplits.load(VASPDataTags, DEFAULTFILEHANDLER)


# %%
