"""Create the external data directory structure used locally or in Colab."""

import argparse
from pathlib import Path


DIRECTORIES = (
    "raw",
    "interim",
    "processed",
    "masks",
    "features",
    "predictions",
    "checkpoints",
    "logs",
    "reports",
    "reports/figures",
    "reports/metrics",
)


def setup_data_directories(data_root: str | Path) -> list[Path]:
    """Create and return all required directories below the data root."""
    root = Path(data_root).expanduser().resolve(strict=False)
    created = []
    for relative_directory in DIRECTORIES:
        directory = root / relative_directory
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="External data root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = setup_data_directories(args.data_root)
    print(f"Data root: {Path(args.data_root).expanduser().resolve(strict=False)}")
    for directory in directories:
        print(f"Ready: {directory}")


if __name__ == "__main__":
    main()
