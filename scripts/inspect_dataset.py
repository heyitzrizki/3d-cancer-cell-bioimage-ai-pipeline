"""Print a tabular summary of a Cell Tracking Challenge dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.data import summarize_dataset_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="CTC dataset directory.")
    return parser.parse_args()


def main() -> None:
    dataset_dir = parse_args().dataset_dir.expanduser().resolve(strict=False)
    summary = summarize_dataset_structure(dataset_dir)
    print(f"Dataset path: {dataset_dir}")
    if summary.empty:
        print("No CTC sequence or ground-truth folders were detected.")
        return
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
