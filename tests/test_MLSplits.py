from omnixas.data.ml_data import MLData, MLSplits


import numpy as np


import json
import unittest
from tempfile import NamedTemporaryFile


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
            data_loaded = json.loads(f.read())
            data_loaded = MLSplits(**data_loaded)
            self.assertIsInstance(data_loaded.train, MLData)
            self.assertIsInstance(data_loaded.val, MLData)
            self.assertIsInstance(data_loaded.test, MLData)
