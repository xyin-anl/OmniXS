# %%
from typing import List

import numpy as np
from pydantic import BaseModel, field_validator, model_serializer


class MLData(BaseModel):
    X: np.ndarray
    y: np.ndarray

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

    class Config:
        arbitrary_types_allowed = True


# %%

if __name__ == "__main__":
    import unittest

    class TestMLData(unittest.TestCase):  # TODO: move to tests
        def test_data(self):
            X = np.random.rand(10, 10)
            y = np.random.rand(10)

            data = MLData(X=X, y=y)
            self.assertIsInstance(data.X, np.ndarray)
            self.assertIsInstance(data.y, np.ndarray)

            data = MLData(X=X.tolist(), y=y.tolist())
            self.assertIsInstance(data.X, np.ndarray)

            # use temp file
            from tempfile import NamedTemporaryFile

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

    unittest.main(argv=[""], exit=False)

    # # %%

# %%
