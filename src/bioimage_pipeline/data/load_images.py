"""Microscopy image loading."""

from pathlib import Path

import numpy as np
import tifffile


def load_tiff(image_path: str | Path) -> np.ndarray:
    """Load a TIFF image or volume into a NumPy array."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {path}")
    return tifffile.imread(path)
