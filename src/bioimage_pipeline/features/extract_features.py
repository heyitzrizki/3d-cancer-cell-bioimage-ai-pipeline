"""Region-level morphology and intensity features."""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def extract_region_features(
    label_image: np.ndarray,
    intensity_image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return basic region properties as a tabular feature set."""
    properties = ("label", "area", "centroid")
    if intensity_image is not None:
        properties += ("intensity_mean",)
    table = regionprops_table(
        np.asarray(label_image),
        intensity_image=None if intensity_image is None else np.asarray(intensity_image),
        properties=properties,
    )
    return pd.DataFrame(table)
