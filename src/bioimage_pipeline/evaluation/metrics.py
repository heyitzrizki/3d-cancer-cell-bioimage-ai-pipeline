"""Binary segmentation metrics."""

import numpy as np


def _binary_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_bool = np.asarray(prediction, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    if prediction_bool.shape != target_bool.shape:
        raise ValueError("Prediction and target must have the same shape.")
    return prediction_bool, target_bool


def dice_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute the Sørensen-Dice coefficient for two binary masks."""
    prediction_bool, target_bool = _binary_pair(prediction, target)
    denominator = prediction_bool.sum() + target_bool.sum()
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(prediction_bool, target_bool).sum()
    return float(2 * intersection / denominator)


def intersection_over_union(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute intersection over union for two binary masks."""
    prediction_bool, target_bool = _binary_pair(prediction, target)
    union = np.logical_or(prediction_bool, target_bool).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(prediction_bool, target_bool).sum()
    return float(intersection / union)
