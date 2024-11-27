from omnixas.data.ml_data import MLData


import numpy as np


import json
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