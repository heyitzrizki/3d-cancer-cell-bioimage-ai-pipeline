"""Print a concise summary of a Cell Tracking Challenge dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.data.inspect_dataset import summarize_dataset_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="CTC dataset directory.")
    return parser.parse_args()


def main() -> None:
    summary = summarize_dataset_structure(parse_args().dataset_dir)
    print(f"Dataset path: {summary['dataset_path']}")

    print("Detected sequences:")
    if not summary["sequences"]:
        print("  None")
    for sequence in summary["sequences"]:
        print(f"  {sequence['name']}: {sequence['tiff_count']} TIFF files")

    print("Detected ground-truth folders:")
    if not summary["ground_truth"]:
        print("  None")
    for ground_truth in summary["ground_truth"]:
        print(f"  {ground_truth['path']}")
        print(f"    SEG: {ground_truth['seg_dir'] or 'not found'}")
        print(f"    TRA: {ground_truth['tra_dir'] or 'not found'}")


if __name__ == "__main__":
    main()
