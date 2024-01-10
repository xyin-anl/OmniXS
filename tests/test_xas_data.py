import os
import unittest

# import pytorch_lightning as pl
import lightning as pl
import torch

from src.data.pl_data import XASData
from utils.src.optuna.dynamic_fc import PlDynamicFC


class TestXASDataModule(unittest.TestCase):
    def test_load_dataset(self):
        query = {
            "compound": "Cu-O",
            "simulation_type": "FEFF",
            "split": "material",
        }
        data_module = XASData(query=query)
        self.assertEqual(data_module.train_dataset[0][0].shape, torch.Size([64]))
        self.assertEqual(data_module.train_dataset[0][1].shape, torch.Size([200]))


class TestXASDataModuleWithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = PlDynamicFC(widths=[64, 400], output_size=200)

    def test_forward(self):
        x = torch.randn(1, 64)
        output = self.model(x)
        self.assertEqual(output.shape, torch.Size([1, 200]))

    def test_testing(self):
        query = {
            "compound": "Cu-O",
            "simulation_type": "FEFF",
            "split": "material",
        }
        data_module = XASData(query=query)
        trainer = pl.Trainer(max_epochs=10)
        trainer.test(self.model, datamodule=data_module)
        self.assertTrue(trainer.logged_metrics["test_loss"] is not None)


if __name__ == "__main__":
    unittest.main()
