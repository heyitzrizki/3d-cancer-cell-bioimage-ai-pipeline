"""Inspect common Cell Tracking Challenge directory structures."""

from pathlib import Path

import pandas as pd

from .load_images import find_tiff_files

SUMMARY_COLUMNS = (
    "dataset",
    "sequence",
    "folder_type",
    "folder_path",
    "exists",
    "tiff_count",
)


def _as_existing_directory(directory: str | Path) -> Path:
    path = Path(directory).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {path}")
    return path


def list_ctc_sequences(raw_dataset_dir: str | Path) -> list[Path]:
    """Return numeric image-sequence directories such as 01 and 02."""
    root = _as_existing_directory(raw_dataset_dir)
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )


def find_ground_truth_dirs(raw_dataset_dir: str | Path) -> list[Path]:
    """Return ground-truth directories such as 01_GT and 02_GT."""
    root = _as_existing_directory(raw_dataset_dir)
    return sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_dir()
        and path.name.endswith("_GT")
        and path.name.removesuffix("_GT").isdigit()
    )


def find_annotation_dirs(ground_truth_dir: str | Path) -> dict[str, Path | None]:
    """Find SEG and TRA directories inside one ground-truth directory."""
    gt_dir = _as_existing_directory(ground_truth_dir)
    return {
        annotation: path.resolve() if (path := gt_dir / annotation).is_dir() else None
        for annotation in ("SEG", "TRA")
    }


def _count_tiff_files(directory: Path | None) -> int:
    if directory is None or not directory.is_dir():
        return 0
    return len(find_tiff_files(directory, recursive=False))


def summarize_dataset_structure(raw_dataset_dir: str | Path) -> pd.DataFrame:
    """Return a tabular summary of CTC sequence and annotation folders."""
    root = _as_existing_directory(raw_dataset_dir)
    rows: list[dict[str, object]] = []

    sequence_names = {path.name for path in list_ctc_sequences(root)}
    gt_dirs = {path.name.removesuffix("_GT"): path for path in find_ground_truth_dirs(root)}
    sequence_names.update(gt_dirs)

    for sequence_name in sorted(sequence_names):
        sequence_dir = root / sequence_name
        rows.append(
            {
                "dataset": root.name,
                "sequence": sequence_name,
                "folder_type": "sequence",
                "folder_path": sequence_dir.resolve(),
                "exists": sequence_dir.is_dir(),
                "tiff_count": _count_tiff_files(sequence_dir if sequence_dir.is_dir() else None),
            }
        )

        gt_dir = gt_dirs.get(sequence_name)
        if gt_dir is None:
            continue

        rows.append(
            {
                "dataset": root.name,
                "sequence": sequence_name,
                "folder_type": "GT",
                "folder_path": gt_dir,
                "exists": True,
                "tiff_count": _count_tiff_files(gt_dir),
            }
        )

        annotation_dirs = find_annotation_dirs(gt_dir)
        for annotation in ("SEG", "TRA"):
            annotation_dir = annotation_dirs[annotation]
            expected_path = gt_dir / annotation
            rows.append(
                {
                    "dataset": root.name,
                    "sequence": sequence_name,
                    "folder_type": annotation,
                    "folder_path": annotation_dir or expected_path.resolve(),
                    "exists": annotation_dir is not None,
                    "tiff_count": _count_tiff_files(annotation_dir),
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
