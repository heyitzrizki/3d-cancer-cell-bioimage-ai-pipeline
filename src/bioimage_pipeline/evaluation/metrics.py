"""Binary and object-count segmentation metrics."""

import numpy as np


def _binary_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_mask = np.asarray(y_true) != 0
    predicted_mask = np.asarray(y_pred) != 0
    if true_mask.shape != predicted_mask.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return true_mask, predicted_mask


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Sørensen-Dice coefficient for binary foreground masks."""
    true_mask, predicted_mask = _binary_pair(y_true, y_pred)
    denominator = true_mask.sum() + predicted_mask.sum()
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(true_mask, predicted_mask).sum()
    return float(2 * intersection / denominator)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute intersection over union for binary foreground masks."""
    true_mask, predicted_mask = _binary_pair(y_true, y_pred)
    union = np.logical_or(true_mask, predicted_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(true_mask, predicted_mask).sum()
    return float(intersection / union)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute binary precision, recall, and F1 with NaN for undefined values."""
    true_mask, predicted_mask = _binary_pair(y_true, y_pred)
    true_positive = np.logical_and(true_mask, predicted_mask).sum()
    false_positive = np.logical_and(~true_mask, predicted_mask).sum()
    false_negative = np.logical_and(true_mask, ~predicted_mask).sum()

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        float(true_positive / precision_denominator)
        if precision_denominator
        else float("nan")
    )
    recall = (
        float(true_positive / recall_denominator)
        if recall_denominator
        else float("nan")
    )
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def object_count_error(y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> int:
    """Return the absolute difference between nonzero object-label counts."""
    true_labels = np.asarray(y_true_labels)
    predicted_labels = np.asarray(y_pred_labels)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("y_true_labels and y_pred_labels must have the same shape.")

    true_count = np.count_nonzero(np.unique(true_labels))
    predicted_count = np.count_nonzero(np.unique(predicted_labels))
    return int(abs(int(predicted_count) - int(true_count)))


def intersection_over_union(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Backward-compatible alias for :func:`iou_score`."""
    return iou_score(y_true, y_pred)
