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
