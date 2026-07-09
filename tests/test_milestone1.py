"""Tests for Milestone 1 dataset inspection and visualization."""

from pathlib import Path

import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bioimage_pipeline.data import (
    find_tiff_files,
    get_image_statistics,
    load_tiff_image,
    summarize_dataset_structure,
)
from bioimage_pipeline.visualization import (
    max_intensity_projection,
    plot_slice,
    save_figure,
)


def test_tiff_loading_and_statistics(tmp_path: Path) -> None:
    image = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    image_path = tmp_path / "sample.TIFF"
    tifffile.imwrite(image_path, image)

    assert find_tiff_files(tmp_path) == [image_path.resolve()]
    loaded = load_tiff_image(image_path)
    statistics = get_image_statistics(loaded)

    assert statistics["shape"] == (2, 3, 4)
    assert statistics["dtype"] == "uint16"
    assert statistics["min_intensity"] == 0.0
    assert statistics["max_intensity"] == 23.0


def test_dataset_summary_handles_missing_annotations(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "01"
    seg_dir = tmp_path / "01_GT" / "SEG"
    sequence_dir.mkdir()
    seg_dir.mkdir(parents=True)
    tifffile.imwrite(sequence_dir / "t000.tif", np.zeros((3, 4), dtype=np.uint8))
    tifffile.imwrite(seg_dir / "man_seg000.tif", np.zeros((3, 4), dtype=np.uint8))

    summary = summarize_dataset_structure(tmp_path)

    assert list(summary["folder_type"]) == ["sequence", "GT", "SEG", "TRA"]
    assert summary.loc[summary["folder_type"] == "SEG", "tiff_count"].item() == 1
    assert not summary.loc[summary["folder_type"] == "TRA", "exists"].item()


def test_projection_plot_and_save(tmp_path: Path) -> None:
    volume = np.arange(24).reshape(2, 3, 4)
    projection = max_intensity_projection(volume)
    assert projection.shape == (3, 4)

    figure, axis = plt.subplots()
    assert plot_slice(projection, ax=axis) is axis
    output_path = save_figure(figure, "projection.png", tmp_path)
    plt.close(figure)

    assert output_path.is_file()
