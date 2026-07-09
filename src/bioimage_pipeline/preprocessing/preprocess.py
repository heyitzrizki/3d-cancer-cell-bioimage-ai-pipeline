"""Preprocessing utilities for fluorescence microscopy images."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian


def _as_supported_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D image or 3D volume, received shape {array.shape}.")
    if array.size == 0:
        raise ValueError("Cannot preprocess an empty image.")
    return array.astype(np.float32, copy=False)


def _spatial_sigma(sigma: float | tuple[float, ...], ndim: int) -> float | tuple[float, ...]:
    """Validate scalar or explicit per-axis Gaussian sigma values."""
    if np.isscalar(sigma):
        sigma_value = float(sigma)
        if sigma_value < 0:
            raise ValueError("sigma must be non-negative.")
        return sigma_value

    sigma_values = tuple(float(value) for value in sigma)
    if len(sigma_values) != ndim:
        raise ValueError(f"sigma must contain {ndim} values for a {ndim}D image.")
    if any(value < 0 for value in sigma_values):
        raise ValueError("sigma values must be non-negative.")
    return sigma_values


def normalize_intensity(
    image: np.ndarray,
    method: str = "percentile",
    lower: float = 1,
    upper: float = 99,
) -> np.ndarray:
    """Normalize image intensities to the [0, 1] interval."""
    array = _as_supported_image(image)
    method_name = method.lower()

    if method_name == "percentile":
        if not 0 <= lower < upper <= 100:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
        low_value, high_value = np.percentile(array, (lower, upper))
    elif method_name in {"minmax", "min-max"}:
        low_value, high_value = float(array.min()), float(array.max())
    else:
        raise ValueError("Normalization method must be 'percentile' or 'minmax'.")

    if high_value <= low_value:
        return np.zeros_like(array, dtype=np.float32)

    normalized = (array - low_value) / (high_value - low_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def denoise_image(
    image: np.ndarray,
    method: str = "gaussian",
    sigma: float | tuple[float, ...] = 1.0,
) -> np.ndarray:
    """Denoise a 2D image or 3D volume while preserving its shape."""
    array = _as_supported_image(image)
    if method.lower() != "gaussian":
        raise ValueError("Denoising method must be 'gaussian'.")
    filter_sigma = _spatial_sigma(sigma, array.ndim)
    if np.all(np.asarray(filter_sigma) == 0):
        return array.copy()

    denoised = gaussian(array, sigma=filter_sigma, preserve_range=True)
    return np.asarray(denoised, dtype=np.float32)


def subtract_background(
    image: np.ndarray,
    sigma: float | tuple[float, ...] = 10.0,
) -> np.ndarray:
    """Estimate smooth background with a Gaussian filter and subtract it."""
    array = _as_supported_image(image)
    filter_sigma = _spatial_sigma(sigma, array.ndim)
    if np.any(np.asarray(filter_sigma) < 0) or not np.any(np.asarray(filter_sigma) > 0):
        raise ValueError("At least one sigma value must be greater than zero.")

    background = ndi.gaussian_filter(array, sigma=filter_sigma)
    corrected = np.maximum(array - background, 0.0)
    return corrected.astype(np.float32, copy=False)


def preprocess_volume(
    image: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Run configurable normalization, denoising, and background subtraction."""
    settings: dict[str, Any] = {
        "normalization_method": "percentile",
        "lower": 1,
        "upper": 99,
        "denoise_method": "gaussian",
        "denoise_sigma": 1.0,
        "subtract_background": True,
        "background_sigma": 10.0,
    }
    if config:
        settings.update(config)

    processed = normalize_intensity(
        image,
        method=settings["normalization_method"],
        lower=float(settings["lower"]),
        upper=float(settings["upper"]),
    )
    processed = denoise_image(
        processed,
        method=settings["denoise_method"],
        sigma=settings["denoise_sigma"],
    )
    if settings["subtract_background"]:
        processed = subtract_background(
            processed,
            sigma=settings["background_sigma"],
        )
        processed = normalize_intensity(processed, method="minmax")

    return processed.astype(np.float32, copy=False)


def percentile_normalize(
    image: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.8,
) -> np.ndarray:
    """Backward-compatible wrapper for percentile normalization."""
    return normalize_intensity(image, method="percentile", lower=lower, upper=upper)
