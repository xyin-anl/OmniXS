# %%
from pydantic import BaseModel, Field, field_serializer
from typing import Any, ClassVar, Dict, List, Optional

# skip splits serilziation
from pydantic import model_serializer


import numpy as np
from pydantic import Field, field_validator

from refactor.data.constants import FEFFDataTags, VASPDataTags
from refactor.data.ml_data import DataTag, MLSplits, MLData

# from refactor.data.constants import FEFFDataTags, VASPDataTags
from refactor.utils import DEFAULTFILEHANDLER
from refactor.utils.io import DEFAULTFILEHANDLER, FileHandler


class MergedSplits(MLSplits):
    splits: Dict[DataTag, MLSplits] = Field(default_factory=dict)

    @classmethod
    def load(
        cls,
        tags: List[DataTag],
        file_handler: "FileHandler",
        balanced: bool = False,
        **kwargs,
    ) -> "MergedSplits":
        splits = [
            file_handler.deserialize_json(MLSplits, supplemental_info=tag)
            for tag in tags
        ]

        if balanced:
            min_sizes = dict(
                train=min(split.train.X.shape[0] for split in splits),
                val=min(split.val.X.shape[0] for split in splits),
                test=min(split.test.X.shape[0] for split in splits),
            )
            splits = [
                MLSplits(
                    train=split.train.shuffled_view()[: min_sizes["train"]],
                    val=split.val.shuffled_view()[: min_sizes["val"]],
                    test=split.test.shuffled_view()[: min_sizes["test"]],
                )
                for split in splits
            ]

        merged = cls()
        for split, tag in zip(splits, tags):
            merged.append(tag, split)
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
    balanced: ClassVar[bool] = False

    def __new__(cls, *args, **kwargs):
        merged = MergedSplits.load(
            FEFFDataTags,
            DEFAULTFILEHANDLER,
            balanced=cls.balanced,
            **kwargs,
        )
        ml_split = MLSplits(
            train=merged.train,
            val=merged.val,
            test=merged.test,
        )
        # return ml_split.shuffled_view()
        return ml_split


class BALANCEDFEFFSplits(FEFFSplits):
    balanced: ClassVar[bool] = True
