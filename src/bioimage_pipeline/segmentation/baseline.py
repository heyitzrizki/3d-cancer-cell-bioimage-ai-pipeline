"""Beginner-friendly classical segmentation baselines."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed


def _as_supported_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D image or 3D volume, received shape {array.shape}.")
    if array.size == 0:
        raise ValueError("Cannot segment an empty image.")
    return array


def otsu_segmentation(image: np.ndarray) -> np.ndarray:
    """Create a binary foreground mask using Otsu's global threshold."""
    array = _as_supported_image(image)
    if np.all(array == array.flat[0]):
        return np.zeros(array.shape, dtype=bool)
    return np.asarray(array > threshold_otsu(array), dtype=bool)


def _valid_block_size(block_size: int, shape: tuple[int, int]) -> int:
    if block_size < 3:
        raise ValueError("block_size must be at least 3.")
    size = min(block_size, *shape)
    if size % 2 == 0:
        size -= 1
    if size < 3:
        raise ValueError("Image dimensions are too small for adaptive thresholding.")
    return size


def adaptive_threshold_segmentation(
    image: np.ndarray,
    block_size: int = 51,
    offset: float = 0,
) -> np.ndarray:
    """Apply adaptive thresholding to a 2D image or each slice of a 3D volume."""
    array = _as_supported_image(image)

    def segment_slice(image_slice: np.ndarray) -> np.ndarray:
        size = _valid_block_size(block_size, image_slice.shape)
        local_threshold = threshold_local(image_slice, block_size=size, offset=offset)
        return image_slice > local_threshold

    if array.ndim == 2:
        return np.asarray(segment_slice(array), dtype=bool)
    return np.stack([segment_slice(image_slice) for image_slice in array]).astype(bool)


def clean_binary_mask(mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    """Remove small foreground regions and fill small holes."""
    binary = np.asarray(mask) != 0
    if binary.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D mask, received shape {binary.shape}.")
    if min_size < 0:
        raise ValueError("min_size must be non-negative.")
    if min_size == 0:
        return binary

    cleaned = remove_small_objects(binary, min_size=min_size)
    cleaned = remove_small_holes(cleaned, area_threshold=min_size)
    return np.asarray(cleaned, dtype=bool)


def label_components(mask: np.ndarray) -> np.ndarray:
    """Assign a positive integer label to each connected foreground component."""
    binary = np.asarray(mask) != 0
    if binary.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D mask, received shape {binary.shape}.")
    return label(binary, connectivity=binary.ndim).astype(np.int32, copy=False)


def watershed_segmentation(
    image: np.ndarray,
    min_distance: int = 5,
    min_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Split touching foreground regions with distance-transform watershed."""
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1.")

    foreground = clean_binary_mask(otsu_segmentation(image), min_size=min_size)
    if not foreground.any():
        empty_labels = np.zeros(foreground.shape, dtype=np.int32)
        return foreground, empty_labels

    distance = ndi.distance_transform_edt(foreground)
    peak_coordinates = peak_local_max(
        distance,
        labels=foreground,
        min_distance=min_distance,
        exclude_border=False,
    )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if peak_coordinates.size:
        markers[tuple(peak_coordinates.T)] = np.arange(1, len(peak_coordinates) + 1)
    else:
        markers = label_components(foreground)

    labels = watershed(-distance, markers, mask=foreground)
    return labels > 0, labels.astype(np.int32, copy=False)


def segment_volume_baseline(
    image: np.ndarray,
    method: str = "otsu",
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment an image and return its binary and connected-component masks."""
    settings: dict[str, Any] = {
        "min_size": 64,
        "block_size": 51,
        "offset": 0,
        "min_distance": 5,
    }
    if config:
        settings.update(config)

    method_name = method.lower()
    if method_name == "otsu":
        binary = otsu_segmentation(image)
        binary = clean_binary_mask(binary, min_size=int(settings["min_size"]))
        labels = label_components(binary)
    elif method_name in {"adaptive", "adaptive_threshold"}:
        binary = adaptive_threshold_segmentation(
            image,
            block_size=int(settings["block_size"]),
            offset=float(settings["offset"]),
        )
        binary = clean_binary_mask(binary, min_size=int(settings["min_size"]))
        labels = label_components(binary)
    elif method_name == "watershed":
        binary, labels = watershed_segmentation(
            image,
            min_distance=int(settings["min_distance"]),
            min_size=int(settings["min_size"]),
        )
    else:
        raise ValueError("method must be 'otsu', 'adaptive', or 'watershed'.")

    return np.asarray(binary, dtype=bool), np.asarray(labels, dtype=np.int32)


def otsu_segment(image: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for :func:`otsu_segmentation`."""
    return otsu_segmentation(image)
