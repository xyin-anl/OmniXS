# %%
from typing import Optional
from refactor.utils import FileHandler

import numpy as np
from pydantic import BaseModel, field_validator, model_serializer

from refactor.data import Element, SpectrumType
from refactor.utils.readable_enums import ReadableEnums


class MLData(BaseModel):
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    def shuffled_view(self, seed: Optional[int] = 42):
        if self.X is None or self.y is None:
            raise ValueError("X and y must be set before shuffling.")
        np.random.seed(seed)
        indices = np.random.permutation(len(self.X))
        return MLData(X=self.X[indices], y=self.y[indices])

    @field_validator("X", "y", mode="before")
    @classmethod
    def _to_numpy(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v

    @model_serializer
    def serialize(self):
        return {
            "X": self.X.tolist(),
            "y": self.y.tolist(),
        }

    def __len__(self):
        len_X = len(self.X) if self.X is not None else 0
        len_y = len(self.y) if self.y is not None else 0
        if len_X != len_y:
            raise ValueError(
                f"X and y must have the same length. Got {len_X} and {len_y}."
            )
        return len_X

    class Config:
        arbitrary_types_allowed = True


@ReadableEnums()
class DataTag(BaseModel):
    element: Element
    type: SpectrumType

    def __hash__(self) -> int:  # store as dict key
        return hash((self.element, self.type))


class MLSplits(BaseModel):
    train: Optional[MLData] = None
    val: Optional[MLData] = None
    test: Optional[MLData] = None

    def shuffled_view(self, seed: Optional[int] = 42):
        return MLSplits(
            train=self.train.shuffled_view(seed),
            val=self.val.shuffled_view(seed),
            test=self.test.shuffled_view(seed),
        )

    def __len__(self):
        return sum(
            len(getattr(self, split_name)) for split_name in self.__fields__.keys()
        )


# %%

if __name__ == "__main__":
    import unittest
    from tempfile import NamedTemporaryFile

    class TestMLData(unittest.TestCase):  # TODO: move to tests
        def test_data(self):
            X = np.random.rand(10, 10)
            y = np.random.rand(10)

            data = MLData(X=X, y=y)
            self.assertIsInstance(data.X, np.ndarray)
            self.assertIsInstance(data.y, np.ndarray)

            data = MLData(X=X.tolist(), y=y.tolist())
            self.assertIsInstance(data.X, np.ndarray)

            temp_file = NamedTemporaryFile()
            with open(temp_file.name, "w") as f:
                f.write(data.json())

            with open(temp_file.name, "r") as f:
                import json

                data_loaded = json.loads(f.read())
                data_loaded = MLData(**data_loaded)
                self.assertIsInstance(data_loaded.X, np.ndarray)
                self.assertTrue(np.allclose(data_loaded.X, data.X))
                self.assertIsInstance(data_loaded.y, np.ndarray)
                self.assertTrue(np.allclose(data_loaded.y, data.y))

        def test_len(self):
            X = np.random.rand(10, 10)
            y = np.random.rand(10)

            data = MLData(X=X, y=y)
            self.assertEqual(len(data), 10)

        def test_len_mismatch(self):
            X = np.random.rand(10, 10)
            y = np.random.rand(11)

            data = MLData(X=X, y=y)
            with self.assertRaises(ValueError):
                len(data)

    class TestMLSplits(unittest.TestCase):
        def test_data(self):
            X = np.random.rand(10, 10)
            y = np.random.rand(10)

            data = MLData(X=X, y=y)
            splits = MLSplits(train=data, val=data, test=data)
            self.assertIsInstance(splits.train, MLData)
            self.assertIsInstance(splits.val, MLData)
            self.assertIsInstance(splits.test, MLData)

            # use temp file

            temp_file = NamedTemporaryFile()
            with open(temp_file.name, "w") as f:
                f.write(splits.json())

            with open(temp_file.name, "r") as f:
                import json

                data_loaded = json.loads(f.read())
                data_loaded = MLSplits(**data_loaded)
                self.assertIsInstance(data_loaded.train, MLData)
                self.assertIsInstance(data_loaded.val, MLData)
                self.assertIsInstance(data_loaded.test, MLData)

    unittest.main(argv=[""], exit=False)


# %%


# %%
