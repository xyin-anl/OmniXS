import os
import unittest

# import pytorch_lightning as pl
import lightning as pl
import torch

from src.data.pl_data import XASData
from utils.src.optuna.dynamic_fc import PlDynamicFC
from src.data.pl_data import DataQuery


class TestXASDataModule(unittest.TestCase):
    def test_load_dataset(self):
        query = DataQuery(
            compound="Cu",
            simulation_type="FEFF",
        )
        data_module = XASData(query=query)
        self.assertEqual(data_module.train_dataset[0][0].shape, torch.Size([64]))


class TestXASDataModuleWithModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # data
        cls.data = XASData(
            DataQuery(
                compound="Cu",
                simulation_type="FEFF",
            )
        )
        # model
        cls.input_size = len(cls.data.train_dataset[0][0])
        cls.output_size = len(cls.data.train_dataset[0][1])
        cls.model = PlDynamicFC(
            widths=[cls.input_size, 400],
            output_size=cls.output_size,
        )

    def test_forward(self):
        x = torch.randn(1, 64)
        output = self.model(x)
        self.assertEqual(output.shape, torch.Size([1, self.output_size]))

    def test_testing(self):
        pl_data = self.data
        trainer = pl.Trainer(max_epochs=10)
        trainer.test(self.model, datamodule=pl_data)
        self.assertTrue(trainer.logged_metrics["test_loss"] is not None)


if __name__ == "__main__":
    unittest.main()
