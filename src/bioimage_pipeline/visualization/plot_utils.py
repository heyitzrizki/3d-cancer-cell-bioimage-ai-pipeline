"""Matplotlib helpers for microscopy images and volumes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_slice(
    image: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot one two-dimensional image slice."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"plot_slice expects a 2D image, received shape {array.shape}.")

    axis = ax if ax is not None else plt.subplots()[1]
    axis.imshow(array, cmap="gray")
    axis.set_axis_off()
    if title:
        axis.set_title(title)
    return axis


def max_intensity_projection(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """Create a max intensity projection from a 3D volume.

    A 2D image is returned unchanged so callers can handle both image types.
    """
    array = np.asarray(volume)
    if array.ndim == 2:
        return array.copy()
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or 3D volume, received shape {array.shape}."
        )
    return np.max(array, axis=axis)


def plot_max_intensity_projection(
    volume: np.ndarray,
    axis: int = 0,
    ax: Axes | None = None,
    title: str = "Max Intensity Projection",
) -> Axes:
    """Create and plot a max intensity projection."""
    projection = max_intensity_projection(volume, axis=axis)
    return plot_slice(projection, ax=ax, title=title)


def save_figure(
    figure: Figure,
    filename: str | Path,
    output_dir: str | Path = Path("reports") / "figures",
    dpi: int = 150,
) -> Path:
    """Save a Matplotlib figure and return the resolved output path."""
    directory = Path(output_dir).expanduser().resolve(strict=False)
    directory.mkdir(parents=True, exist_ok=True)

    name = Path(filename)
    if name.name != str(name):
        raise ValueError("filename must be a file name without parent directories.")
    output_path = directory / name
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def _display_plane(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[array.shape[0] // 2]
    raise ValueError(f"Expected a 2D image or 3D volume, received shape {array.shape}.")


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a binary mask over a 2D image or middle slice of a 3D volume."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    image_plane = _display_plane(image)
    mask_plane = _display_plane(mask) != 0
    if image_plane.shape != mask_plane.shape:
        raise ValueError("Image and mask display planes must have the same shape.")

    axis = ax if ax is not None else plt.subplots()[1]
    axis.imshow(image_plane, cmap="gray")
    overlay = np.ma.masked_where(~mask_plane, mask_plane.astype(np.float32))
    axis.imshow(overlay, cmap="Reds", alpha=alpha, vmin=0, vmax=1)
    axis.set_axis_off()
    if title:
        axis.set_title(title)
    return axis


def save_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Save a segmentation overlay figure."""
    path = Path(output_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 7))
    overlay_mask_on_image(image, mask, ax=axis, title=title)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_side_by_side_segmentation(
    raw_image: np.ndarray,
    processed_image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save raw, processed, and overlay views in one figure."""
    raw_plane = _display_plane(raw_image)
    processed_plane = _display_plane(processed_image)
    mask_plane = _display_plane(mask)
    if raw_plane.shape != processed_plane.shape or raw_plane.shape != mask_plane.shape:
        raise ValueError("Raw image, processed image, and mask must align.")

    path = Path(output_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_slice(raw_plane, ax=axes[0], title="Raw")
    plot_slice(processed_plane, ax=axes[1], title="Preprocessed")
    overlay_mask_on_image(raw_plane, mask_plane, ax=axes[2], title="Segmentation")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
