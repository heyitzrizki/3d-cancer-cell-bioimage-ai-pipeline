"""Classical microscopy preprocessing."""

import numpy as np


def percentile_normalize(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.8,
) -> np.ndarray:
    """Scale intensities between two percentiles to the [0, 1] interval."""
    if not 0 <= lower < upper <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    image_float = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image_float, (lower, upper))
    if high <= low:
        return np.zeros_like(image_float)
    return np.clip((image_float - low) / (high - low), 0.0, 1.0)
