import os
import unittest
from src.data.feff_data_raw import RAWDataFEFF
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


class TestRAWDataFEFF(unittest.TestCase):
    def setUp(self):
        self.data = RAWDataFEFF(compound="Ti")

    def test_parameters(self):
        parameters = self.data.parameters
        self.assertIsInstance(parameters, dict)
        for key, value in parameters.items():
            self.assertIsInstance(key, tuple)
            self.assertIsInstance(value, dict)
            self.assertIn("mu", value)
            self.assertIsInstance(value["mu"], np.ndarray)

    def test_missing_data(self):
        self.assertIsInstance(self.data.missing_data, set)
        for item in self.data.missing_data:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_material_ids(self):
        id = next(iter(self.data._material_ids))
        self.assertIsInstance(id, str)

    def test_sites(self):
        id = next(iter(self.data._material_ids))
        site = next(iter(self.data._sites[id]))
        self.assertIsInstance(site, str)

    def test_plot(self):
        id = ("mp-390", "000_Ti")  # reference to another paper data

        plt.plot(
            self.data.parameters[id]["mu"][:, 0],
            self.data.parameters[id]["mu"][:, 1],
        )

        save_dir = os.path.join("tests", "test_plots")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"raw_data_feff_{id[0]}_{id[1]}_{date_time}.png"
        plt.savefig(os.path.join(save_dir, file_name))

        plt.cla()


if __name__ == "__main__":
    unittest.main()
