# %%
from typing import Any, Self

from pydantic import Field, model_validator
from sklearn.preprocessing import RobustScaler

from refactor.data import MLData, MLSplits


class ScaledMlSplit(MLSplits):
    x_scaler: Any = Field(default_factory=RobustScaler)
    y_scaler: Any = Field(default_factory=RobustScaler)

    @model_validator(mode="after")
    def fit_transform(self):
        if not self.train:
            raise ValueError("train data is required")
        self.fit(self.train)._self_transform()
        return self

    def fit(self, split: MLData) -> Self:
        self.x_scaler.fit(split.X)
        self.y_scaler.fit(split.y)
        return self

    def _self_transform(self) -> Self:
        self.train = self.transform(self.train)
        self.test = self.transform(self.test) if self.test else None
        self.val = self.transform(self.val) if self.val else None
        return self

    def transform(self, data: MLData) -> MLData:
        return MLData(
            X=self.x_scaler.transform(data.X),
            y=self.y_scaler.transform(data.y),
        )

    def inverse_transform(self, data: MLData) -> MLData:
        return MLData(
            X=self.x_scaler.inverse_transform(data.X),
            y=self.y_scaler.inverse_transform(data.y),
        )

    @classmethod
    def from_splits(cls, splits: MLSplits):
        return cls(train=splits.train, test=splits.test, val=splits.val)


# %%

if __name__ == "__main__":
    import numpy as np

    from refactor.data.merge_ml_splits import FEFFSplits

    split = FEFFSplits
    scaled_mlsplit = ScaledMlSplit.from_splits(split)

    print(f"Original data: {split.train.X[0][:3]}")
    print(f"Scaled data: {scaled_mlsplit.train.X[0][:3]}")
    print(
        f"Inverted data: {scaled_mlsplit.inverse_transform(scaled_mlsplit.train).X[0][:3]}"
    )

# %%
