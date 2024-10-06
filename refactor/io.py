# %%
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Type

import yaml
from pydantic import BaseModel

from refactor.spectra_data import (
    FEFF,
    Element,
    ElementSpectra,
    Material,
    Site,
    SpectraType,
    Spectrum,
)


class FileHandler:
    def __init__(self, config_path: str):
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def save(
        self,
        obj: BaseModel,
        obj_type: str,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ):
        config = self.config.get(obj_type)
        if not config:
            raise ValueError(f"Configuration for {obj_type} not found")

        dir_path = config["directory"]
        os.makedirs(dir_path, exist_ok=True)

        filename = self._get_filename(obj, config)
        filepath = os.path.join(dir_path, filename)

        # Merge default and provided include/exclude
        default_include = set(config.get("include", []))
        default_exclude = set(config.get("exclude", []))
        final_include = (
            default_include.union(include) if include else default_include or None
        )
        final_exclude = (
            default_exclude.union(exclude) if exclude else default_exclude or None
        )

        serialization_method = config.get("serialization", "json")
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
        obj_type: str,
        model: Type[BaseModel],
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        **kwargs,
    ) -> BaseModel:
        config = self.config.get(obj_type)
        if not config:
            raise ValueError(f"Configuration for {obj_type} not found")

        dir_path = config["directory"]
        filename = self._get_filename_for_load(kwargs, config)
        filepath = os.path.join(dir_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Merge default and provided include/exclude
        default_include = set(config.get("include", []))
        default_exclude = set(config.get("exclude", []))
        final_include = (
            default_include.union(include) if include else default_include or None
        )
        final_exclude = (
            default_exclude.union(exclude) if exclude else default_exclude or None
        )

        serialization_method = config.get("serialization", "json")
        if serialization_method == "json":
            with open(filepath, "r") as f:
                data = json.load(f)

            # Apply include/exclude filters
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
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj

    def _get_filename(self, obj: Any, config: Dict[str, Any]) -> str:
        filename_template = config["filename"]
        properties = self._get_args(filename_template)

        for p in properties:
            try:
                value = self._get_nested_attr(obj, p)
                filename_template = filename_template.replace("{" + p + "}", str(value))
            except AttributeError:
                raise ValueError(f"Unable to access attribute '{p}' in the object")

        return filename_template

    def _get_filename_for_load(
        self, kwargs: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        filename_template = config["filename"]
        properties = self._get_args(filename_template)

        for p in properties:
            parts = p.split(".")
            value = kwargs
            try:
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    elif hasattr(value, part):
                        value = getattr(value, part)
                    else:
                        # If the attribute is not found, we'll use an empty string
                        # This allows for optional nested attributes
                        value = ""
                        break
                filename_template = filename_template.replace("{" + p + "}", str(value))
            except AttributeError:
                # If we can't access an attribute, we'll use an empty string
                filename_template = filename_template.replace("{" + p + "}", "")

        return filename_template


if __name__ == "__main__":
    # Initialize the FileHandler
    file_handler = FileHandler("refactor/path.yaml")

    feff_spectrum = Spectrum(
        type=SpectraType.FEFF,
        energies=[1, 2, 3],
        intensities=[4, 5, 6],
    )
    material1 = Material(
        id="mp-1234",
        site=Site(index=0, element=Element.Ti, spectra={FEFF: feff_spectrum}),
    )

    file_handler.save(material1, "Material")

    material2 = Material(
        id="mp-1235",
        site=Site(index=0, element=Element.Ti, spectra={FEFF: feff_spectrum}),
    )

    element_spectra = ElementSpectra(
        element=Element.Ti,
        type=SpectraType.FEFF,
        materials=[material1, material2],
    )

    file_handler.save(element_spectra, "ElementSpectra")

    loaded_material = file_handler.load(
        "Material", Material, id="mp-1234", site={"index": 0}
    )

    # Load an ElementSpectra object
    loaded_element_spectra = file_handler.load(
        "ElementSpectra", ElementSpectra, element="Ti", type="FEFF"
    )

    # Now you can use the loaded objects
    print(f"Loaded material : {loaded_material}")
    print(f"Loaded element spectra : {loaded_element_spectra}")

# %%
