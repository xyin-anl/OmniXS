# %%
import warnings
from functools import partial
from typing import Any, Self

from pydantic import Field, model_validator
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from omnixas.data import MLData, MLSplits


class UncenteredRobustScaler(RobustScaler):
    """For spectra prediction models that produce only positive values (e.g XASBlock)"""

    def __init__(self):
        super().__init__(with_centering=False)


class IdentityScaler(FunctionTransformer):
    """Useful for cleaner code later"""

    def __init__(self):
        super().__init__(
            func=lambda X: X,
            inverse_func=lambda X: X,
        )


class MultiplicativeScaler(FunctionTransformer):
    def __init__(self, factor: float):
        self.factor = factor
        super().__init__(
            func=lambda X: X * self.factor,
            inverse_func=lambda X: X / self.factor,
        )


class ThousandScaler(FunctionTransformer):
    """As done in arxiv_v1 manuscript"""

    def __init__(self):
        super().__init__(
            func=lambda X: X * 1000,
            inverse_func=lambda X: X / 1000,
        )


class ScaledMlSplit(MLSplits):
    x_scaler: Any = Field(default_factory=IdentityScaler)
    y_scaler: Any = Field(default_factory=IdentityScaler)

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
        self.test = self.transform(self.test)
        self.val = self.transform(self.val)
        return self

    def transform(self, data: MLData) -> MLData:
        split = MLData(
            X=self.x_scaler.transform(data.X),
            y=self.y_scaler.transform(data.y),
        )
        if (split.y < 0).any():
            msg = "y values are negative after scaling."
            msg += "Avoid using models that only gives positive values"
            msg += "e.g: XASBlock with softplus activation"
            warnings.warn(msg)
        return split

    def inverse_transform(self, data: MLData) -> MLData:
        return MLData(
            X=self.x_scaler.inverse_transform(data.X),
            y=self.y_scaler.inverse_transform(data.y),
        )

    @classmethod
    def from_splits(
        cls,
        splits: MLSplits,
        x_scaler: Any = ThousandScaler,
        y_scaler: Any = ThousandScaler,
    ):
        return cls(
            train=splits.train,
            test=splits.test,
            val=splits.val,
            x_scaler=x_scaler(),
            y_scaler=y_scaler(),
        )


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    from omnixas.data.merge_ml_splits import FEFFSplits

    split = FEFFSplits

    # scaled_mlsplit = ScaledMlSplit.from_splits(
    #     split, scaler=RobustScaler
    # )  # should raise warning

    # scaled_mlsplit = ScaledMlSplit.from_splits(split, scaler=ThousadScaler)

    # scaled_mlsplit = ScaledMlSplit.from_splits(
    #     split, scaler=partial(MultiplicativeScaler, factor=10)
    # )  # warning: hydra do not support partial

    scaled_mlsplit = ScaledMlSplit.from_splits(
        split,
        scaler=UncenteredRobustScaler,
    )

    # plt.hist(scaled_mlsplit.test.y.flatten(), bins=100)

    print(f"Original y data: {split.train.y[0][:3]}")
    print(f"Scaled y data: {scaled_mlsplit.train.y[0][:3]}")
    print(
        f"Inverted y data: {scaled_mlsplit.inverse_transform(scaled_mlsplit.train).y[0][:3]}"
    )


# %%
