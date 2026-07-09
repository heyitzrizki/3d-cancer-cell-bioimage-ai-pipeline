"""Smoke tests for the initial package structure."""

from pathlib import Path

import bioimage_pipeline
from bioimage_pipeline.config import load_yaml_config
from bioimage_pipeline.data import summarize_dataset_structure
from bioimage_pipeline.evaluation import dice_coefficient
from bioimage_pipeline.features import extract_region_features
from bioimage_pipeline.preprocessing import percentile_normalize
from bioimage_pipeline.segmentation import otsu_segment
from bioimage_pipeline.utils import get_project_root
from bioimage_pipeline.visualization import plot_slice


def test_package_imports() -> None:
    assert bioimage_pipeline.__version__ == "0.1.0"
    assert callable(load_yaml_config)
    assert callable(summarize_dataset_structure)
    assert callable(dice_coefficient)
    assert callable(extract_region_features)
    assert callable(percentile_normalize)
    assert callable(otsu_segment)
    assert callable(plot_slice)


def test_project_root() -> None:
    root = get_project_root()
    assert isinstance(root, Path)
    assert (root / "pyproject.toml").is_file()
