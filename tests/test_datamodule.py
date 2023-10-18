import os
import unittest

# import pytorch_lightning as pl
import lightning as pl
import torch

from src.dataset import XASDataModule
from utils.src.optuna.dynamic_fc import PlDynamicFC


class TestXASDataModule(unittest.TestCase):
    def test_load_dataset(self):
        compound = "Cu-O"
        data_dir = os.path.join(
            "dataset/ML-231009",
            f"{compound}_K-edge_FEFF_XANES",
            "material-splits",
            "data",
        )

        data_module = XASDataModule(data_dir=data_dir)
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
        compound = "Cu-O"
        data_dir = os.path.join(
            "dataset/ML-231009",
            f"{compound}_K-edge_FEFF_XANES",
            "material-splits",
            "data",
        )
        data_module = XASDataModule(data_dir=data_dir)
        trainer = pl.Trainer(max_epochs=10)
        trainer.test(self.model, datamodule=data_module)
        self.assertTrue(trainer.logged_metrics["test_loss"] is not None)


if __name__ == "__main__":
    unittest.main()
