"""Inspect common Cell Tracking Challenge directory structures."""

from pathlib import Path
from typing import Any


def _as_existing_directory(directory: str | Path) -> Path:
    path = Path(directory).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {path}")
    return path


def find_tiff_files(directory: str | Path) -> list[Path]:
    """Return TIFF files below a directory, sorted by path."""
    path = _as_existing_directory(directory)
    files = [*path.rglob("*.tif"), *path.rglob("*.tiff")]
    return sorted({file.resolve() for file in files})


def list_ctc_sequences(raw_dataset_dir: str | Path) -> list[Path]:
    """Return numeric CTC image-sequence directories such as 01 and 02."""
    root = _as_existing_directory(raw_dataset_dir)
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def find_ground_truth_dirs(raw_dataset_dir: str | Path) -> list[Path]:
    """Return CTC ground-truth directories such as 01_GT and 02_GT."""
    root = _as_existing_directory(raw_dataset_dir)
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir()
        and path.name.endswith("_GT")
        and path.name.removesuffix("_GT").isdigit()
    )


def summarize_dataset_structure(raw_dataset_dir: str | Path) -> dict[str, Any]:
    """Summarize sequences, TIFF counts, and available SEG/TRA annotations."""
    root = _as_existing_directory(raw_dataset_dir)
    sequence_summaries = [
        {
            "name": sequence.name,
            "path": sequence,
            "tiff_count": len(find_tiff_files(sequence)),
        }
        for sequence in list_ctc_sequences(root)
    ]

    ground_truth = []
    for gt_dir in find_ground_truth_dirs(root):
        seg_dir = gt_dir / "SEG"
        tra_dir = gt_dir / "TRA"
        ground_truth.append(
            {
                "name": gt_dir.name,
                "path": gt_dir,
                "seg_dir": seg_dir.resolve() if seg_dir.is_dir() else None,
                "tra_dir": tra_dir.resolve() if tra_dir.is_dir() else None,
            }
        )

    return {
        "dataset_path": root,
        "sequences": sequence_summaries,
        "ground_truth": ground_truth,
    }
