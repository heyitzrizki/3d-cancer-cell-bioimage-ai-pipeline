"""Validate MVP configuration and describe the planned classical pipeline."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML file.")
    return parser.parse_args()


def main() -> None:
    config = load_yaml_config(parse_args().config)
    experiment = config.get("experiment", {}).get("name", "unnamed")
    dataset = config.get("dataset", {}).get("name", "unspecified")
    print(f"Configuration valid: experiment={experiment}, dataset={dataset}")
    print("MVP execution will be added after data exploration validates preprocessing choices.")


if __name__ == "__main__":
    main()
