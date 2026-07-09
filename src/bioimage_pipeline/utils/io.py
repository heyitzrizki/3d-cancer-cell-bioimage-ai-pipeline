"""Small file I/O helpers."""

from pathlib import Path
from typing import Any

import yaml


def write_yaml(data: dict[str, Any], output_path: str | Path) -> Path:
    """Write a mapping as YAML."""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return path.resolve(strict=False)
