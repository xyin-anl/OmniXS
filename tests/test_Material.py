from refactor.spectra_data import Material


from matgl.ext.pymatgen import Structure
from pydantic import ValidationError


import unittest
from unittest.mock import Mock


class TestMaterial(unittest.TestCase):
    def test_valid_material(self):
        material = Material(id="mp-1234", structure=Mock(spec=Structure))
        self.assertEqual(material.id, "mp-1234")
        self.assertIsNotNone(material.structure)

    def test_invalid_material_id(self):
        with self.assertRaises(ValidationError):
            Material(id="invalid-id")

    def test_empty_material_id(self):
        with self.assertRaises(ValidationError):
            Material(id="")


if __name__ == "__main__":
    unittest.main()
