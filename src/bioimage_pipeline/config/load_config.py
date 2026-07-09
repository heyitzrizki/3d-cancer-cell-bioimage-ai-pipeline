"""YAML configuration loading."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its top-level mapping."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping at the top level: {path}")
    return config
