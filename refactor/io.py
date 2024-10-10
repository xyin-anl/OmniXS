# %%
import json
import os
import re
import yaml
from typing import Any, Dict, List, Optional, Set, Type, Union
from pydantic import BaseModel, FilePath

YAMLCONFIGPATH = FilePath


class FileHandler:
    def __init__(
        self,
        config: Union[YAMLCONFIGPATH, Dict[str, Any]],
        default_serialization: str = "json",
        replace_existing: bool = False,
    ):
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Configuration file not found: {config}")
            with open(config, "r") as config_file:
                self.config = yaml.safe_load(config_file)
        else:
            self.config = config
        self.default_serialization = default_serialization
        self.replace_existing = replace_existing

    def save(
        self,
        obj: BaseModel,
        config_name: Union[str, None] = None,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ):
        if config_name is None:
            config_name = obj.__class__.__name__
        config = self.config.get(config_name)
        if not config:
            raise ValueError(f"Configuration for {config_name} not found")

        dir_path = config["directory"]
        dir_path = self._resolve_template(obj, dir_path)
        os.makedirs(dir_path, exist_ok=True)

        filename = self._resolve_template(obj, config["filename"])
        filepath = os.path.join(dir_path, filename)

        if os.path.exists(filepath) and not self.replace_existing:
            raise FileExistsError(f"File already exists: {filepath}")

        default_include = set(config.get("include", []))
        default_exclude = set(config.get("exclude", []))
        final_include = (
            default_include.union(include) if include else default_include or None
        )
        final_exclude = (
            default_exclude.union(exclude) if exclude else default_exclude or None
        )

        serialization_method = config.get("serialization", self.default_serialization)
        if serialization_method == "json":
            with open(filepath, "w") as f:
                json.dump(
                    obj.model_dump(include=final_include, exclude=final_exclude),
                    f,
                    indent=2,
                )
        else:
            raise ValueError(
                f"Unsupported serialization method: {serialization_method}"
            )

    def load(
        self,
        model: Type[BaseModel],
        config_name: Union[str, None] = None,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        **kwargs,
    ) -> BaseModel:
        if config_name is None:
            config_name = model.__name__
        config = self.config.get(config_name)
        if not config:
            raise ValueError(f"Configuration for {config_name} not found")

        dir_path = config["directory"]
        dir_path = self._resolve_load_template(kwargs, dir_path)

        filename = self._resolve_load_template(kwargs, config["filename"])
        filepath = os.path.join(dir_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        default_include = set(config.get("include", []))
        default_exclude = set(config.get("exclude", []))
        final_include = (
            default_include.union(include) if include else default_include or None
        )
        final_exclude = (
            default_exclude.union(exclude) if exclude else default_exclude or None
        )

        serialization_method = config.get("serialization", self.default_serialization)
        if serialization_method == "json":
            with open(filepath, "r") as f:
                data = json.load(f)

            if final_include:
                data = {k: v for k, v in data.items() if k in final_include}
            if final_exclude:
                data = {k: v for k, v in data.items() if k not in final_exclude}

            return model(**data)
        else:
            raise ValueError(
                f"Unsupported serialization method: {serialization_method}"
            )

    def _get_args(self, filename: str) -> List[str]:
        return re.findall(r"\{(.*?)\}", filename, re.DOTALL)

    def _get_nested_attr(self, obj: Any, attr: str) -> Any:
        def _recursive_get(obj: Any, parts: List[str]) -> Any:
            if not parts:
                return obj
            part = parts[0]
            if isinstance(obj, dict):
                return _recursive_get(obj.get(part, {}), parts[1:])
            elif hasattr(obj, part):
                return _recursive_get(getattr(obj, part), parts[1:])
            else:
                return None

        return _recursive_get(obj, attr.split("."))

    # def _get_filename_for_load( self, kwargs: Dict[str, Any], config: Dict[str, Any]) -> str:
    def _resolve_load_template(self, kwargs: Dict[str, Any], template: str) -> str:
        # filename_template = config["filename"]
        properties = self._get_args(template)

        for p in properties:
            value = self._get_nested_attr(kwargs, p)
            value = "" if value is None else str(value)
            template = template.replace("{" + p + "}", value)

        return template

    # def _get_filename(self, obj: Any, config: Dict[str, Any]) -> str:
    def _resolve_template(self, obj: Any, template: str) -> str:
        # filename_template = config["filename"]
        properties = self._get_args(template)

        for p in properties:
            value = self._get_nested_attr(obj, p)
            if value is None:
                value = ""  # or some default value
            template = template.replace("{" + p + "}", str(value))

        return template


# %%

if __name__ == "__main__":
    config = {
        "Material": {
            "directory": ".",
            "filename": "{id}.json",
            "serialization": "json",
        }
    }
    file_handler = FileHandler(config)

    from tests.test_utils import create_dummy_material, create_dummy_spectrum
    from refactor.spectra_data import Material

    dummy_material = create_dummy_material()
    file_handler.save(create_dummy_material())
    loaded_material = file_handler.load(model=Material, id="mp-1234", site={"index": 0})

# %%
