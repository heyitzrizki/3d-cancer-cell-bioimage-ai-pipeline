"""Interpretable segmentation baselines."""

import numpy as np
from skimage.filters import threshold_otsu


def otsu_segment(image: np.ndarray) -> np.ndarray:
    """Create a binary foreground mask using Otsu's threshold."""
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Cannot segment an empty image.")
    return array > threshold_otsu(array)
