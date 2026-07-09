"""Cell Tracking Challenge ground-truth discovery and frame matching."""

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .load_images import find_tiff_files, load_tiff_image

_GT_FRAME_PATTERN = re.compile(r"man_(?:seg|track)_?(\d+)", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\d+")


def _normalize_sequence_id(sequence_id: str | int) -> str:
    sequence = str(sequence_id)
    return sequence.zfill(2) if sequence.isdigit() else sequence


def find_segmentation_gt_files(
    dataset_dir: str | Path,
    sequence_id: str | int,
) -> list[Path]:
    """Find segmentation ground-truth TIFF files for one sequence."""
    root = Path(dataset_dir).expanduser().resolve(strict=False)
    seg_dir = root / f"{_normalize_sequence_id(sequence_id)}_GT" / "SEG"
    if not seg_dir.is_dir():
        return []
    return [
        path
        for path in find_tiff_files(seg_dir)
        if path.name.lower().startswith("man_seg")
    ]


def find_tracking_gt_files(
    dataset_dir: str | Path,
    sequence_id: str | int,
) -> list[Path]:
    """Find tracking ground-truth TIFF files for one sequence."""
    root = Path(dataset_dir).expanduser().resolve(strict=False)
    tra_dir = root / f"{_normalize_sequence_id(sequence_id)}_GT" / "TRA"
    if not tra_dir.is_dir():
        return []
    return [
        path
        for path in find_tiff_files(tra_dir)
        if path.name.lower().startswith("man_track")
    ]


def _prediction_frame_id(path: Path) -> int | None:
    numbers = _NUMBER_PATTERN.findall(path.stem)
    return int(numbers[0]) if numbers else None


def _gt_frame_id(path: Path) -> int | None:
    match = _GT_FRAME_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def get_segmentation_gt_slice_index(gt_path: str | Path) -> int | None:
    """Return a z-slice index from names such as man_seg_000_013.tif."""
    numbers = _NUMBER_PATTERN.findall(Path(gt_path).stem)
    return int(numbers[1]) if len(numbers) >= 2 else None


def match_prediction_to_gt_frame(
    prediction_files: Iterable[str | Path],
    gt_files: Iterable[str | Path],
) -> dict[Path, list[Path]]:
    """Map each prediction file to all GT files with the same frame index."""
    grouped_gt: dict[int, list[Path]] = {}
    for gt_file in gt_files:
        gt_path = Path(gt_file).expanduser().resolve(strict=False)
        frame_id = _gt_frame_id(gt_path)
        if frame_id is not None:
            grouped_gt.setdefault(frame_id, []).append(gt_path)

    matches: dict[Path, list[Path]] = {}
    for prediction_file in prediction_files:
        prediction_path = Path(prediction_file).expanduser().resolve(strict=False)
        frame_id = _prediction_frame_id(prediction_path)
        if frame_id is not None and frame_id in grouped_gt:
            matches[prediction_path] = sorted(grouped_gt[frame_id])
    return matches


def load_gt_mask(gt_path: str | Path) -> np.ndarray:
    """Load a labeled CTC ground-truth mask."""
    return load_tiff_image(gt_path)
