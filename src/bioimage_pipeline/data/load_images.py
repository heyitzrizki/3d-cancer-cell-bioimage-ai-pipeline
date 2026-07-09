"""TIFF discovery, loading, and image metadata utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

TIFF_SUFFIXES = {".tif", ".tiff"}


def find_tiff_files(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Find TIFF files in a directory, optionally including subdirectories."""
    root = Path(directory).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path.resolve()
        for path in candidates
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def load_tiff_image(image_path: str | Path) -> np.ndarray:
    """Load one .tif or .tiff image as a NumPy array."""
    path = Path(image_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {path}")
    if path.suffix.lower() not in TIFF_SUFFIXES:
        raise ValueError(f"Expected a .tif or .tiff file: {path}")

    try:
        return np.asarray(tifffile.imread(path))
    except (OSError, ValueError, tifffile.TiffFileError) as error:
        if "imagecodecs" in str(error).lower():
            raise RuntimeError(
                f"Could not decode compressed TIFF file {path}. "
                "Install the project dependencies, including imagecodecs."
            ) from error
        raise ValueError(f"Could not read TIFF file: {path}") from error


def get_image_statistics(image: np.ndarray) -> dict[str, Any]:
    """Return shape, dtype, dimensions, and basic intensity statistics."""
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty image.")

    return {
        "shape": tuple(int(size) for size in array.shape),
        "ndim": int(array.ndim),
        "dtype": str(array.dtype),
        "min_intensity": float(np.min(array)),
        "max_intensity": float(np.max(array)),
        "mean_intensity": float(np.mean(array)),
    }


def read_tiff_statistics(image_path: str | Path) -> dict[str, Any]:
    """Load one TIFF file and return its image statistics."""
    return get_image_statistics(load_tiff_image(image_path))


def load_tiff(image_path: str | Path) -> np.ndarray:
    """Backward-compatible alias for :func:`load_tiff_image`."""
    return load_tiff_image(image_path)
