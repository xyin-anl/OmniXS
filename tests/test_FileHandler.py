import unittest
import os
import tempfile
import yaml
from pydantic import BaseModel
from typing import Optional

from refactor.io import FileHandler
from refactor.spectra_data import Material, Spectrum
from tests.test_utils import create_dummy_material, create_dummy_spectrum


class TestModel(BaseModel):
    id: str
    value: int
    optional: Optional[str] = None


class TestNestedModel(BaseModel):
    id: str
    value: int
    nested: TestModel


class TestDoubleNestedModel(BaseModel):
    id: str
    value: int
    nested: TestNestedModel


class TestFileHandler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "TestModel": {
                "directory": self.temp_dir,
                "filename": "{id}.json",
                "serialization": "json",
            },
            "TestNestedModel": {
                "directory": self.temp_dir,
                "filename": "{id}_{nested.id}.json",
                "serialization": "json",
            },
            "TestDoubleNestedModel": {
                "directory": self.temp_dir,
                "filename": "{id}_{nested.id}_{nested.nested.id}.json",
                "serialization": "json",
            },
        }
        self.file_handler = FileHandler(config=self.config)

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_save_and_load_simple_model(self):
        model = TestModel(id="test1", value=42)
        self.file_handler.save(model)
        loaded_model = self.file_handler.load(TestModel, id="test1")
        self.assertEqual(model, loaded_model)

    def test_save_and_load_with_optional_field(self):
        model = TestModel(id="test2", value=42, optional="present")
        self.file_handler.save(model)
        loaded_model = self.file_handler.load(TestModel, id="test2")
        self.assertEqual(model, loaded_model)

    def test_save_with_include(self):
        model = TestModel(id="test3", value=42, optional="include_me")
        self.file_handler.save(model, include={"id", "value", "optional"})
        loaded_model = self.file_handler.load(TestModel, id="test3")
        self.assertEqual(loaded_model.id, "test3")
        self.assertEqual(loaded_model.optional, "include_me")
        self.assertIsNotNone(loaded_model.value)

    def test_save_with_exclude(self):
        model = TestModel(id="test4", value=42, optional="exclude_me")
        self.file_handler.save(model, exclude={"optional"})
        loaded_model = self.file_handler.load(TestModel, id="test4")
        self.assertEqual(loaded_model.id, "test4")
        self.assertEqual(loaded_model.value, 42)
        self.assertIsNone(loaded_model.optional)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            self.file_handler.load(TestModel, id="nonexistent")

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            self.file_handler.save(
                TestModel(id="test5", value=42), config_name="NonexistentModel"
            )

    def test_unsupported_serialization(self):
        config = self.config.copy()
        config["TestModel"]["serialization"] = "unsupported"
        file_handler = FileHandler(config=config)
        with self.assertRaises(ValueError):
            file_handler.save(TestModel(id="test6", value=42))

    def test_yaml_config_file(self):
        config_path = os.path.join(self.temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)
        file_handler = FileHandler(config=config_path)
        model = TestModel(id="test7", value=42)
        file_handler.save(model)
        loaded_model = file_handler.load(TestModel, id="test7")
        self.assertEqual(model, loaded_model)

    def test_save_nested_model(self):
        nested_model = TestModel(id="nested", value=42, optional="nested")
        model = TestNestedModel(id="test8", value=42, nested=nested_model)
        self.file_handler.save(model)

    def test_load_nested_model(self):
        nested_model = TestModel(id="nested", value=42, optional="nested")
        model = TestNestedModel(id="test9", value=42, nested=nested_model)
        self.file_handler.save(model)
        loaded_model = self.file_handler.load(
            TestNestedModel, id="test9", nested={"id": "nested"}
        )
        self.assertEqual(model, loaded_model)

    def test_save_double_nested_model(self):
        double_nested_model = TestDoubleNestedModel(
            id="double_nested",
            value=42,
            nested=TestNestedModel(
                id="nested",
                value=42,
                nested=TestModel(id="nested_nested", value=42),
            ),
        )
        self.file_handler.save(double_nested_model)

    def test_load_double_nested_model(self):
        double_nested_model = TestDoubleNestedModel(
            id="double_nested",
            value=42,
            nested=TestNestedModel(
                id="nested",
                value=42,
                nested=TestModel(id="nested_nested", value=42),
            ),
        )
        self.file_handler.save(double_nested_model)
        loaded_model = self.file_handler.load(
            TestDoubleNestedModel,
            id="double_nested",
            nested={"id": "nested", "nested": {"id": "nested_nested"}},
        )
        self.assertEqual(double_nested_model, loaded_model)


if __name__ == "__main__":
    unittest.main()
