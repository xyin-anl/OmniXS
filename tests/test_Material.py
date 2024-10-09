import unittest

from pydantic import ValidationError

from refactor.io import FileHandler
from refactor.spectra_data import Material
from tests.test_utils import create_dummy_material


class TestMaterial(unittest.TestCase):

    def test_valid_material(self):
        material = create_dummy_material()
        self.assertEqual(material.id, "mp-1234")
        self.assertIsNotNone(material.structure)

    def test_invalid_material_id(self):
        with self.assertRaises(ValidationError):
            invalid_id = "invalid_prefix" + "-1234"
            Material(id=invalid_id)

    def test_empty_material_id(self):
        with self.assertRaises(ValidationError):
            Material(id="")

    def test_serialization(self):
        material = create_dummy_material()
        material.model_dump()

    def test_io(self):
        material = create_dummy_material()
        import yaml

        config = {
            "Material": {
                "directory": ".",
                "filename": "{id}.json",
                "serialization": "json",
            }
        }
        file_handler = FileHandler(config=config)
        file_handler.save(material, "Material")
        loaded_material = file_handler.load(Material, id="mp-1234")
        self.assertEqual(material, loaded_material)


if __name__ == "__main__":
    unittest.main()
