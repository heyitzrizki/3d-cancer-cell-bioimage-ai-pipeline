"""Microscopy plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_slice(image: np.ndarray, ax: Axes | None = None, title: str | None = None) -> Axes:
    """Display a 2D image slice using a grayscale colormap."""
    if np.asarray(image).ndim != 2:
        raise ValueError("plot_slice expects a two-dimensional image.")
    axis = ax if ax is not None else plt.subplots()[1]
    axis.imshow(image, cmap="gray")
    axis.set_axis_off()
    if title:
        axis.set_title(title)
    return axis
