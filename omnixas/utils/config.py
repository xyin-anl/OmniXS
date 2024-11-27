from pathlib import Path
import yaml
import os


class Config:
    def __init__(self, config_path: str = None):
        """Initialize with optional custom config path"""
        self.config_path = Path(
            config_path or os.getenv("OMNIXAS_CONFIG", "config/default.yaml")
        )

    def load(self) -> dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def save(self, config: dict, path: str = None):
        """Save configuration to YAML file"""
        save_path = Path(path or self.config_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            yaml.safe_dump(config, f)
