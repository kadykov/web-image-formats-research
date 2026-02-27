"""Tests for visual comparison module."""

import json
import shutil
import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.comparison import (
    ComparisonConfig,
    ComparisonResult,
    StrategyResult,
    WorstRegion,
    _build_label,
    _build_metric_label,
    _find_worst_region_in_array,
    _get_or_encode,
    _read_pfm,
    _render_distmap_thumbnail,
    _resolve_encoded_path,
    assemble_comparison_grid,
    compute_aggregate_distortion_maps,
    crop_and_zoom,
    determine_varying_parameters,
    encode_image,
    find_worst_measurement,
    find_worst_region,
    find_worst_source_image,
    generate_comparison,
    generate_distortion_map,
    get_worst_image_score,
    load_quality_results,
)

# ---------------------------------------------------------------------------
# PFM helpers
# ---------------------------------------------------------------------------


def _write_pfm_grayscale(path: Path, data: np.ndarray) -> None:
    """Write a grayscale PFM (Pf) file with little-endian float32."""
    height, width = data.shape
    with open(path, "wb") as fh:
        fh.write(b"Pf\n")
        fh.write(f"{width} {height}\n".encode())
        fh.write(b"-1.0\n")  # negative = little-endian
        # PFM stores rows bottom-to-top
        for row in reversed(range(height)):
            fh.write(struct.pack(f"<{width}f", *data[row].astype(np.float32).tolist()))


def _write_pfm_color(path: Path, data: np.ndarray) -> None:
    """Write a colour PFM (PF) file with little-endian float32.

    ``data`` must have shape (H, W, 3).
    """
    height, width = data.shape[:2]
    with open(path, "wb") as fh:
        fh.write(b"PF\n")
        fh.write(f"{width} {height}\n".encode())
        fh.write(b"-1.0\n")
        for row in reversed(range(height)):
            for col in range(width):
                r, g, b = data[row, col]
                fh.write(struct.pack("<3f", float(r), float(g), float(b)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def sample_quality_data() -> dict:
    """Create sample quality measurement data for testing."""
    return {
        "study_id": "test-comparison",
        "study_name": "Test Comparison Study",
        "dataset": {
            "id": "test-dataset",
            "path": "data/datasets/test",
            "image_count": 2,
        },
        "timestamp": "2026-02-20T12:00:00+00:00",
        "measurements": [
            {
                "source_image": "data/datasets/test/image1.png",
                "original_image": "data/datasets/test/image1.png",
                "encoded_path": "data/encoded/test/jpeg/image1_q50.jpg",
                "format": "jpeg",
                "quality": 50,
                "file_size": 50000,
                "width": 256,
                "height": 256,
                "source_file_size": 200000,
                "ssimulacra2": 55.0,
                "psnr": 31.0,
                "ssim": 0.90,
                "butteraugli": 5.5,
                "chroma_subsampling": None,
                "speed": None,
                "effort": None,
                "method": None,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image1.png",
                "original_image": "data/datasets/test/image1.png",
                "encoded_path": "data/encoded/test/jpeg/image1_q80.jpg",
                "format": "jpeg",
                "quality": 80,
                "file_size": 100000,
                "width": 256,
                "height": 256,
                "source_file_size": 200000,
                "ssimulacra2": 75.0,
                "psnr": 38.0,
                "ssim": 0.96,
                "butteraugli": 2.0,
                "chroma_subsampling": None,
                "speed": None,
                "effort": None,
                "method": None,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image1.png",
                "original_image": "data/datasets/test/image1.png",
                "encoded_path": "data/encoded/test/avif/image1_q50.avif",
                "format": "avif",
                "quality": 50,
                "file_size": 40000,
                "width": 256,
                "height": 256,
                "source_file_size": 200000,
                "ssimulacra2": 70.0,
                "psnr": 36.0,
                "ssim": 0.94,
                "butteraugli": 3.0,
                "chroma_subsampling": "420",
                "speed": 6,
                "effort": None,
                "method": None,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image1.png",
                "original_image": "data/datasets/test/image1.png",
                "encoded_path": "data/encoded/test/avif/image1_q80.avif",
                "format": "avif",
                "quality": 80,
                "file_size": 80000,
                "width": 256,
                "height": 256,
                "source_file_size": 200000,
                "ssimulacra2": 85.0,
                "psnr": 42.0,
                "ssim": 0.98,
                "butteraugli": 1.2,
                "chroma_subsampling": "420",
                "speed": 6,
                "effort": None,
                "method": None,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image2.png",
                "original_image": "data/datasets/test/image2.png",
                "encoded_path": "data/encoded/test/jpeg/image2_q50.jpg",
                "format": "jpeg",
                "quality": 50,
                "file_size": 60000,
                "width": 256,
                "height": 256,
                "source_file_size": 250000,
                "ssimulacra2": 50.0,
                "psnr": 29.0,
                "ssim": 0.88,
                "butteraugli": 6.0,
                "chroma_subsampling": None,
                "speed": None,
                "effort": None,
                "method": None,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image2.png",
                "original_image": "data/datasets/test/image2.png",
                "encoded_path": "data/encoded/test/jpeg/image2_q80.jpg",
                "format": "jpeg",
                "quality": 80,
                "file_size": 110000,
                "width": 256,
                "height": 256,
                "source_file_size": 250000,
                "ssimulacra2": 70.0,
                "psnr": 36.0,
                "ssim": 0.94,
                "butteraugli": 2.5,
                "chroma_subsampling": None,
                "speed": None,
                "effort": None,
                "method": None,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
        ],
    }


@pytest.fixture
def quality_json_file(tmp_path: Path, sample_quality_data: dict) -> Path:
    """Write sample quality data to a JSON file."""
    path = tmp_path / "quality.json"
    with open(path, "w") as f:
        json.dump(sample_quality_data, f)
    return path


@pytest.fixture
def sample_rgb_image(tmp_path: Path) -> Path:
    """Create a sample RGB test image (256x256) with some patterns."""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    pixels = img.load()
    # Add some variation - a bright red patch in the center
    for x in range(100, 156):
        for y in range(100, 156):
            assert pixels is not None
            pixels[x, y] = (255, 0, 0)
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


@pytest.fixture
def sample_distortion_map(tmp_path: Path) -> Path:
    """Create a sample raw PFM distortion map.

    Simulates a Butteraugli rawdistmap with a hot region at (80, 60).
    """
    arr = np.zeros((256, 256), dtype=np.float32)
    arr[60:90, 80:110] = 5.0  # hot-spot
    arr[10:30, 10:30] = 1.0  # moderate distortion elsewhere
    path = tmp_path / "distmap.pfm"
    _write_pfm_grayscale(path, arr)
    return path


# ---------------------------------------------------------------------------
# Tests: load_quality_results
# ---------------------------------------------------------------------------


def test_load_quality_results(quality_json_file: Path) -> None:
    """Test loading quality results from JSON."""
    data = load_quality_results(quality_json_file)
    assert "measurements" in data
    assert len(data["measurements"]) == 6
    assert data["study_id"] == "test-comparison"


def test_load_quality_results_not_found() -> None:
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_quality_results(Path("nonexistent.json"))


def test_load_quality_results_no_measurements(tmp_path: Path) -> None:
    """Test that missing measurements field raises ValueError."""
    path = tmp_path / "bad.json"
    with open(path, "w") as f:
        json.dump({"study_id": "test"}, f)
    with pytest.raises(ValueError, match="No 'measurements' field"):
        load_quality_results(path)


# ---------------------------------------------------------------------------
# Tests: find_worst_measurement
# ---------------------------------------------------------------------------


def test_find_worst_measurement_ssimulacra2(sample_quality_data: dict) -> None:
    """Test finding worst measurement by SSIMULACRA2 (lowest score)."""
    worst = find_worst_measurement(sample_quality_data["measurements"], metric="ssimulacra2")
    assert worst["ssimulacra2"] == 50.0  # image2 q50 has the lowest
    assert worst["source_image"] == "data/datasets/test/image2.png"
    assert worst["quality"] == 50


def test_find_worst_measurement_butteraugli(sample_quality_data: dict) -> None:
    """Test finding worst measurement by Butteraugli (highest score)."""
    worst = find_worst_measurement(sample_quality_data["measurements"], metric="butteraugli")
    assert worst["butteraugli"] == 6.0  # image2 q50 has the highest
    assert worst["source_image"] == "data/datasets/test/image2.png"


def test_find_worst_measurement_psnr(sample_quality_data: dict) -> None:
    """Test finding worst measurement by PSNR (lowest score)."""
    worst = find_worst_measurement(sample_quality_data["measurements"], metric="psnr")
    assert worst["psnr"] == 29.0


def test_find_worst_measurement_no_valid(sample_quality_data: dict) -> None:
    """Test that ValueError is raised when no valid measurements exist."""
    # Set all measurements to have errors
    for m in sample_quality_data["measurements"]:
        m["measurement_error"] = "test error"
    with pytest.raises(ValueError, match="No valid measurements"):
        find_worst_measurement(sample_quality_data["measurements"])


def test_find_worst_measurement_none_metric(sample_quality_data: dict) -> None:
    """Test that measurements with None metrics are skipped."""
    for m in sample_quality_data["measurements"]:
        m["ssimulacra2"] = None
    with pytest.raises(ValueError, match="No valid measurements"):
        find_worst_measurement(sample_quality_data["measurements"], metric="ssimulacra2")


# ---------------------------------------------------------------------------
# Tests: find_worst_source_image
# ---------------------------------------------------------------------------


def test_find_worst_source_image_ssimulacra2(sample_quality_data: dict) -> None:
    """Test finding worst source image by average SSIMULACRA2."""
    # image1 avg: (55 + 75 + 70 + 85) / 4 = 71.25
    # image2 avg: (50 + 70) / 2 = 60.0
    worst = find_worst_source_image(sample_quality_data["measurements"], metric="ssimulacra2")
    assert worst == "data/datasets/test/image2.png"


def test_find_worst_source_image_butteraugli(sample_quality_data: dict) -> None:
    """Test finding worst source image by average Butteraugli."""
    # image1 avg: (5.5 + 2.0 + 3.0 + 1.2) / 4 = 2.925
    # image2 avg: (6.0 + 2.5) / 2 = 4.25
    worst = find_worst_source_image(sample_quality_data["measurements"], metric="butteraugli")
    assert worst == "data/datasets/test/image2.png"


def test_find_worst_source_image_variance(sample_quality_data: dict) -> None:
    """Test finding worst source image by SSIMULACRA2 variance."""
    # image1 scores: [55, 75, 70, 85] → mean=71.25, var=((55-71.25)^2+(75-71.25)^2+(70-71.25)^2+(85-71.25)^2)/4 = 112.1875
    # image2 scores: [50, 70] → mean=60, var=((50-60)^2+(70-60)^2)/2 = 100.0
    worst = find_worst_source_image(
        sample_quality_data["measurements"],
        metric="ssimulacra2",
        strategy="variance",
    )
    assert worst == "data/datasets/test/image1.png"


def test_find_worst_source_image_unknown_strategy(sample_quality_data: dict) -> None:
    """Test that unknown strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown image selection strategy"):
        find_worst_source_image(
            sample_quality_data["measurements"],
            strategy="unknown",
        )


def test_get_worst_image_score_average(sample_quality_data: dict) -> None:
    """Test computing average image score."""
    score = get_worst_image_score(
        sample_quality_data["measurements"],
        "data/datasets/test/image2.png",
        metric="ssimulacra2",
        strategy="average",
    )
    assert abs(score - 60.0) < 0.01


def test_get_worst_image_score_variance(sample_quality_data: dict) -> None:
    """Test computing variance image score."""
    score = get_worst_image_score(
        sample_quality_data["measurements"],
        "data/datasets/test/image2.png",
        metric="ssimulacra2",
        strategy="variance",
    )
    assert abs(score - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: find_worst_region
# ---------------------------------------------------------------------------


def test_find_worst_region(sample_distortion_map: Path) -> None:
    """Test finding the worst region in a raw PFM distortion map."""
    region = find_worst_region(sample_distortion_map, crop_size=32)
    # The hot spot is at (80, 60) with size 30x30
    # A 32x32 window should overlap the hot spot
    assert 48 <= region.x <= 110
    assert 28 <= region.y <= 90
    assert region.width == 32
    assert region.height == 32
    assert region.avg_distortion > 0


def test_find_worst_region_small_image(tmp_path: Path) -> None:
    """Test finding worst region when image is smaller than crop size."""
    arr = np.full((64, 64), 3.0, dtype=np.float32)
    path = tmp_path / "small.pfm"
    _write_pfm_grayscale(path, arr)

    region = find_worst_region(path, crop_size=128)
    assert region.x == 0
    assert region.y == 0
    assert region.width == 64
    assert region.height == 64


def test_find_worst_region_uniform(tmp_path: Path) -> None:
    """Test with a uniform map (any region should be equivalent)."""
    arr = np.full((256, 256), 2.5, dtype=np.float32)
    path = tmp_path / "uniform.pfm"
    _write_pfm_grayscale(path, arr)

    region = find_worst_region(path, crop_size=64)
    assert region.width == 64
    assert region.height == 64
    assert abs(region.avg_distortion - 2.5) < 0.01


def test_find_worst_region_pfm(tmp_path: Path) -> None:
    """find_worst_region uses _read_pfm branch for .pfm files."""
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[10:20, 40:50] = 5.0  # hot-spot
    pfm_path = tmp_path / "distmap.pfm"
    _write_pfm_grayscale(pfm_path, arr)

    region = find_worst_region(pfm_path, crop_size=16)
    # Window should capture the hot spot around column 40-50, row 10-20
    assert 24 <= region.x <= 50
    assert 0 <= region.y <= 20
    assert region.width == 16
    assert region.height == 16
    assert region.avg_distortion > 0


# ---------------------------------------------------------------------------
# Tests: _read_pfm
# ---------------------------------------------------------------------------


def test_read_pfm_grayscale(tmp_path: Path) -> None:
    """_read_pfm returns correct values for a Pf (grayscale) PFM file."""
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    pfm_path = tmp_path / "gray.pfm"
    _write_pfm_grayscale(pfm_path, arr)

    result = _read_pfm(pfm_path)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, arr, rtol=1e-5)


def test_read_pfm_color(tmp_path: Path) -> None:
    """_read_pfm returns max-across-channels for PF (colour) PFM files."""
    # Shape (2, 2, 3)
    arr = np.array(
        [[[1.0, 2.0, 0.5], [0.1, 0.2, 0.3]], [[3.0, 1.0, 2.0], [0.0, 0.0, 4.0]]], dtype=np.float32
    )
    pfm_path = tmp_path / "color.pfm"
    _write_pfm_color(pfm_path, arr)

    result = _read_pfm(pfm_path)
    assert result.shape == (2, 2)
    # max across channels per pixel
    expected = np.array([[2.0, 0.3], [3.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_read_pfm_bottom_to_top_flip(tmp_path: Path) -> None:
    """_read_pfm flips rows so the first row in result is the top of the image."""
    # Row 0 has value 10, row 1 has value 20
    arr = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    pfm_path = tmp_path / "flip.pfm"
    _write_pfm_grayscale(pfm_path, arr)

    result = _read_pfm(pfm_path)
    # After flip, top row should still be [10, 10]
    np.testing.assert_allclose(result[0], [10.0, 10.0], rtol=1e-5)
    np.testing.assert_allclose(result[1], [20.0, 20.0], rtol=1e-5)


def test_read_pfm_invalid_magic(tmp_path: Path) -> None:
    """_read_pfm raises ValueError for non-PFM files."""
    bad = tmp_path / "bad.pfm"
    bad.write_bytes(b"P6\n2 2\n255\n" + bytes(12))
    with pytest.raises(ValueError, match="Not a PFM file"):
        _read_pfm(bad)


# ---------------------------------------------------------------------------
# Tests: _find_worst_region_in_array
# ---------------------------------------------------------------------------


def test_find_worst_region_in_array_finds_hotspot() -> None:
    """_find_worst_region_in_array locates the hot-spot in a float array."""
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:30, 40:50] = 10.0  # hot-spot at col 40-50, row 20-30
    region = _find_worst_region_in_array(arr, crop_size=16)
    assert 24 <= region.x <= 50
    assert 4 <= region.y <= 30
    assert region.width == 16
    assert region.height == 16
    assert region.avg_distortion > 0


def test_find_worst_region_in_array_small_image() -> None:
    """Returns whole image when smaller than crop_size."""
    arr = np.full((20, 30), 5.0, dtype=np.float64)
    region = _find_worst_region_in_array(arr, crop_size=64)
    assert region.x == 0
    assert region.y == 0
    assert region.width == 30
    assert region.height == 20
    assert abs(region.avg_distortion - 5.0) < 1e-9


def test_find_worst_region_in_array_uniform() -> None:
    """Any position is equally valid for a uniform array."""
    arr = np.full((128, 128), 3.0, dtype=np.float64)
    region = _find_worst_region_in_array(arr, crop_size=32)
    assert region.width == 32
    assert region.height == 32
    assert abs(region.avg_distortion - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Tests: compute_aggregate_distortion_maps
# ---------------------------------------------------------------------------


def test_compute_aggregate_distortion_maps_unit(
    tmp_path: Path,
) -> None:
    """Unit test: mocks encoder and distortion-map tools to test aggregation.

    Verifies that the function correctly stacks the per-variant arrays,
    and that the results (avg, var) have the expected shapes and values.
    """
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "image2.png"
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(source_path)

    encoded_dir = tmp_path / "encoded"
    encoded_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Two measurements (same format, different quality) for image2
    measurements = [
        {"source_image": "image2.png", "format": "jpeg", "quality": 50, "encoded_path": ""},
        {"source_image": "image2.png", "format": "jpeg", "quality": 80, "encoded_path": ""},
    ]

    # Synthetic PFM arrays that the mock will write
    arr1 = np.full((8, 8), 4.0, dtype=np.float32)
    arr2 = np.full((8, 8), 2.0, dtype=np.float32)

    def fake_gen_distortion_map(
        _original: Path,
        _compressed: Path,
        output_pfm: Path,
    ) -> Path:
        # Write a synthetic PFM so _read_pfm can load it
        arr = arr1 if "q50" in str(output_pfm) else arr2
        _write_pfm_grayscale(output_pfm, arr)
        return output_pfm

    fake_enc = tmp_path / "fake.jpg"
    fake_enc.write_bytes(b"fake")

    with (
        patch("src.comparison._get_or_encode", return_value=fake_enc),
        patch("src.comparison.generate_distortion_map", side_effect=fake_gen_distortion_map),
    ):
        avg_map, var_map, variant_pairs = compute_aggregate_distortion_maps(
            source_path,
            measurements,
            output_dir,
            project_root=tmp_path,
            encoded_dir=encoded_dir,
        )

    assert avg_map.shape == (8, 8)
    assert var_map.shape == (8, 8)
    # Average of 4.0 and 2.0 = 3.0
    np.testing.assert_allclose(avg_map, 3.0, rtol=1e-5)
    # Variance of [4.0, 2.0] = 1.0
    np.testing.assert_allclose(var_map, 1.0, rtol=1e-5)
    # Both measurements should have produced arrays
    assert len(variant_pairs) == 2
    arrs = [a for a, _ in variant_pairs]
    assert all(a.shape == (8, 8) for a in arrs)


def test_compute_aggregate_distortion_maps_no_maps(
    tmp_path: Path,
) -> None:
    """RuntimeError is raised when every variant fails."""
    with (
        patch("src.comparison._get_or_encode", return_value=None),
        pytest.raises(RuntimeError, match="No distortion maps"),
    ):
        compute_aggregate_distortion_maps(
            source_path=tmp_path / "img.png",
            measurements=[{"format": "jpeg", "quality": 50, "encoded_path": ""}],
            output_dir=tmp_path / "out",
            project_root=tmp_path,
            encoded_dir=tmp_path / "enc",
        )


# ---------------------------------------------------------------------------
# Tests: crop_and_zoom
# ---------------------------------------------------------------------------


def test_crop_and_zoom(sample_rgb_image: Path, tmp_output: Path) -> None:
    """Test cropping and zooming a region."""
    region = WorstRegion(x=100, y=100, width=56, height=56, avg_distortion=100.0)
    output_path = tmp_output / "crop.png"

    result = crop_and_zoom(sample_rgb_image, region, zoom_factor=2, output_path=output_path)

    assert output_path.exists()
    assert result.size == (112, 112)  # 56 * 2 = 112


def test_crop_and_zoom_no_save(sample_rgb_image: Path) -> None:
    """Test cropping and zooming without saving."""
    region = WorstRegion(x=0, y=0, width=64, height=64, avg_distortion=50.0)
    result = crop_and_zoom(sample_rgb_image, region, zoom_factor=3)
    assert result.size == (192, 192)  # 64 * 3 = 192


def test_crop_and_zoom_clamps_to_bounds(sample_rgb_image: Path) -> None:
    """Test that crop region is clamped to image bounds."""
    region = WorstRegion(x=200, y=200, width=128, height=128, avg_distortion=50.0)
    result = crop_and_zoom(sample_rgb_image, region, zoom_factor=2)
    # Image is 256x256, so from (200, 200) we can only get 56x56
    assert result.size == (112, 112)  # 56 * 2


# ---------------------------------------------------------------------------
# Tests: determine_varying_parameters
# ---------------------------------------------------------------------------


def test_determine_varying_parameters(sample_quality_data: dict) -> None:
    """Test detecting varying parameters."""
    varying = determine_varying_parameters(sample_quality_data["measurements"])
    assert "format" in varying
    assert "quality" in varying


def test_determine_varying_parameters_single_format() -> None:
    """Test with single format (only quality varies)."""
    measurements = [
        {"format": "avif", "quality": 50, "speed": 6},
        {"format": "avif", "quality": 80, "speed": 6},
    ]
    varying = determine_varying_parameters(measurements)
    assert "quality" in varying
    assert "format" not in varying
    assert "speed" not in varying


# ---------------------------------------------------------------------------
# Tests: label building
# ---------------------------------------------------------------------------


def test_build_label() -> None:
    """Test building a descriptive label."""
    m = {"format": "avif", "quality": 50, "speed": 3, "chroma_subsampling": "420"}
    label = _build_label(m, ["format", "quality", "speed"])
    assert "AVIF" in label
    assert "q50" in label
    assert "speed=3" in label


def test_build_label_format_only() -> None:
    """Test label with only format varying."""
    m = {"format": "jpeg", "quality": 80}
    label = _build_label(m, ["format"])
    assert label == "JPEG"


def test_build_metric_label() -> None:
    """Test building a metric label."""
    m = {"ssimulacra2": 75.5, "butteraugli": 2.5, "file_size": 51200}
    label = _build_metric_label(m)
    assert "SSIM2:75.5" in label
    assert "BA:2.50" in label
    assert "50KB" in label


def test_build_metric_label_missing() -> None:
    """Test metric label with missing values."""
    m = {"ssimulacra2": None, "butteraugli": None, "file_size": None}
    label = _build_metric_label(m)
    assert label == ""


# ---------------------------------------------------------------------------
# Tests: assemble_comparison_grid (requires ImageMagick)
# ---------------------------------------------------------------------------


@pytest.fixture
def crop_images(tmp_path: Path) -> list[tuple[Path, str, str]]:
    """Create small test images for grid assembly."""
    crops = []
    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(i * 60, 100, 100))
        path = tmp_path / f"crop_{i}.png"
        img.save(path)
        crops.append((path, f"Format q{50 + i * 10}", f"SSIM2:{70 + i:.1f}"))
    return crops


@pytest.mark.skipif(
    not shutil.which("montage"),
    reason="ImageMagick montage not available",
)
def test_assemble_comparison_grid(
    crop_images: list[tuple[Path, str, str]],
    tmp_output: Path,
) -> None:
    """Test assembling a comparison grid."""
    output_path = tmp_output / "grid.png"
    result = assemble_comparison_grid(crop_images, output_path, max_columns=4)

    assert result == output_path
    assert output_path.exists()

    with Image.open(output_path) as img:
        # Grid should have been created
        assert img.width > 0
        assert img.height > 0


@pytest.mark.skipif(
    not shutil.which("montage"),
    reason="ImageMagick montage not available",
)
def test_assemble_comparison_grid_wraps_rows(
    tmp_path: Path,
    tmp_output: Path,
) -> None:
    """Test that grid wraps to multiple rows when exceeding max_columns."""
    crops = []
    for i in range(8):
        img = Image.new("RGB", (64, 64), color=(i * 30, 100, 100))
        path = tmp_path / f"crop_{i}.png"
        img.save(path)
        crops.append((path, f"Test {i}", f"Score: {i}"))

    output_path = tmp_output / "wide_grid.png"
    result = assemble_comparison_grid(crops, output_path, max_columns=4)
    assert result.exists()


# ---------------------------------------------------------------------------
# Tests: generate_distortion_map (requires butteraugli_main)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not shutil.which("butteraugli_main"),
    reason="butteraugli_main not available",
)
def test_generate_distortion_map(tmp_path: Path) -> None:
    """Test generating a raw Butteraugli PFM distortion map."""
    # Create original and a slightly different compressed version
    original = Image.new("RGB", (64, 64), color=(128, 128, 128))
    original_path = tmp_path / "original.png"
    original.save(original_path)

    compressed = Image.new("RGB", (64, 64), color=(130, 126, 132))
    compressed_path = tmp_path / "compressed.png"
    compressed.save(compressed_path)

    output_pfm = tmp_path / "distmap.pfm"
    result = generate_distortion_map(original_path, compressed_path, output_pfm)

    assert result == output_pfm
    assert output_pfm.exists()

    # PFM should be readable and contain float values
    arr = _read_pfm(output_pfm)
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests: _render_distmap_thumbnail
# ---------------------------------------------------------------------------


def test_render_distmap_thumbnail_output_shape(tmp_path: Path) -> None:
    """Thumbnail is always exactly target_size×target_size regardless of source shape."""
    arr = np.full((200, 300), 5.0, dtype=np.float64)  # non-square source
    out = tmp_path / "thumb.png"
    _render_distmap_thumbnail(arr, target_size=384, output_path=out)

    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (384, 384)


def test_render_distmap_thumbnail_fixed_scale(tmp_path: Path) -> None:
    """Pixels at vmax map to the brightest viridis colour; vmin maps to darkest."""
    zero_arr = np.zeros((16, 16), dtype=np.float64)
    out_zero = tmp_path / "zero.png"
    _render_distmap_thumbnail(zero_arr, target_size=16, output_path=out_zero, vmax=10.0)

    vmax_arr = np.full((16, 16), 10.0, dtype=np.float64)
    out_vmax = tmp_path / "vmax.png"
    _render_distmap_thumbnail(vmax_arr, target_size=16, output_path=out_vmax, vmax=10.0)

    with Image.open(out_zero) as img:
        zero_px = np.array(img)
    with Image.open(out_vmax) as img:
        vmax_px = np.array(img)

    # With viridis_r: zero distortion → bright yellow, vmax distortion → dark purple.
    # So zero pixels are brighter than vmax pixels.
    assert np.all(zero_px == zero_px[0, 0]), "Zero arr should produce uniform colour"
    assert np.all(vmax_px == vmax_px[0, 0]), "vmax arr should produce uniform colour"
    # The average brightness of zero-distortion pixels should be higher than vmax pixels
    assert float(zero_px.mean()) > float(vmax_px.mean())


def test_render_distmap_thumbnail_clamps_out_of_range(tmp_path: Path) -> None:
    """Values above vmax are clamped to the brightest colour, not wrapped."""
    at_vmax = np.full((16, 16), 10.0, dtype=np.float64)
    above_vmax = np.full((16, 16), 100.0, dtype=np.float64)

    out_at = tmp_path / "at_vmax.png"
    out_above = tmp_path / "above_vmax.png"
    _render_distmap_thumbnail(at_vmax, target_size=16, output_path=out_at, vmax=10.0)
    _render_distmap_thumbnail(above_vmax, target_size=16, output_path=out_above, vmax=10.0)

    with Image.open(out_at) as img:
        px_at = np.array(img)
    with Image.open(out_above) as img:
        px_above = np.array(img)

    np.testing.assert_array_equal(px_at, px_above)


def test_render_distmap_thumbnail_consistent_scale(tmp_path: Path) -> None:
    """Two different arrays use same scale — higher values produce brighter pixels."""
    arr_low = np.full((16, 16), 1.0, dtype=np.float64)
    arr_high = np.full((16, 16), 9.0, dtype=np.float64)

    out_low = tmp_path / "low.png"
    out_high = tmp_path / "high.png"
    _render_distmap_thumbnail(arr_low, target_size=16, output_path=out_low, vmax=10.0)
    _render_distmap_thumbnail(arr_high, target_size=16, output_path=out_high, vmax=10.0)

    with Image.open(out_low) as img:
        mean_low = float(np.array(img).mean())
    with Image.open(out_high) as img:
        mean_high = float(np.array(img).mean())

    # With viridis_r: lower distortion → brighter pixels, higher → darker.
    assert mean_low > mean_high


# ---------------------------------------------------------------------------
# Tests: ComparisonConfig and ComparisonResult
# ---------------------------------------------------------------------------


def test_comparison_config_defaults() -> None:
    """Test default configuration values."""
    config = ComparisonConfig()
    assert config.crop_size == 128
    assert config.zoom_factor == 3
    assert config.metric == "ssimulacra2"
    assert config.max_columns == 4
    assert config.label_font_size == 22
    assert config.strategy == "both"
    assert config.distmap_vmax == 5.0
    assert config.source_image is None


def test_comparison_config_strategies() -> None:
    """Test all supported strategy values."""
    for strategy in ("average", "variance", "both"):
        config = ComparisonConfig(strategy=strategy)
        assert config.strategy == strategy


def test_comparison_config_custom() -> None:
    """Test custom configuration values."""
    config = ComparisonConfig(crop_size=96, zoom_factor=3, metric="butteraugli")
    assert config.crop_size == 96
    assert config.zoom_factor == 3
    assert config.metric == "butteraugli"


def test_comparison_config_source_image_override() -> None:
    """Test source_image override."""
    config = ComparisonConfig(source_image="data/datasets/test/image1.png")
    assert config.source_image == "data/datasets/test/image1.png"


def test_comparison_result() -> None:
    """Test ComparisonResult dataclass."""
    region = WorstRegion(x=10, y=20, width=128, height=128, avg_distortion=150.0)
    sr = StrategyResult(
        strategy="average",
        source_image="image.png",
        image_score=60.0,
        worst_format="jpeg",
        worst_quality=50,
        worst_metric_value=55.0,
        region=region,
        output_dir=Path("comparison/average"),
        output_images=[Path("comparison.png")],
    )
    result = ComparisonResult(
        study_id="test",
        strategies=[sr],
        varying_parameters=["format", "quality"],
    )
    assert result.study_id == "test"
    assert len(result.strategies) == 1
    assert result.strategies[0].region.x == 10
    assert result.strategies[0].strategy == "average"


def test_strategy_result() -> None:
    """StrategyResult records all per-strategy fields."""
    region = WorstRegion(x=0, y=0, width=64, height=64, avg_distortion=1.0)
    sr = StrategyResult(
        strategy="variance",
        source_image="img.png",
        image_score=15.0,
        worst_format="avif",
        worst_quality=60,
        worst_metric_value=60.0,
        region=region,
        output_dir=Path("comparison/variance"),
    )
    assert sr.strategy == "variance"
    assert sr.image_score == 15.0
    assert sr.output_images == []


# ---------------------------------------------------------------------------
# Tests: full generate_comparison (integration, requires tools)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not shutil.which("butteraugli_main") or not shutil.which("montage"),
    reason="butteraugli_main and/or montage not available",
)
def test_generate_comparison_integration(
    tmp_path: Path,
    sample_quality_data: dict,
) -> None:
    """Integration test for the full comparison pipeline.

    Creates real test source images and runs the full pipeline which
    re-encodes images on the fly (no pre-existing encoded files needed).
    """
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Create source images (the only images that need to exist on disk)
    img_dir = project_root / "data" / "datasets" / "test"
    img_dir.mkdir(parents=True)

    img1 = Image.new("RGB", (256, 256), color=(128, 128, 128))
    # Add some texture to make encoding differences visible
    pixels1 = img1.load()
    assert pixels1 is not None
    for x in range(0, 256, 2):
        for y in range(0, 256, 2):
            pixels1[x, y] = (130, 126, 132)
    img1_path = img_dir / "image1.png"
    img1.save(img1_path)

    img2 = Image.new("RGB", (256, 256), color=(200, 100, 50))
    pixels2 = img2.load()
    assert pixels2 is not None
    for x in range(0, 256, 3):
        for y in range(0, 256, 3):
            pixels2[x, y] = (210, 90, 60)
    img2_path = img_dir / "image2.png"
    img2.save(img2_path)

    # Use only JPEG measurements (always available, no special decoder needed)
    jpeg_only_data = {
        **sample_quality_data,
        "measurements": [m for m in sample_quality_data["measurements"] if m["format"] == "jpeg"],
    }

    # Write quality.json
    quality_path = tmp_path / "quality.json"
    with open(quality_path, "w") as f:
        json.dump(jpeg_only_data, f)

    output_dir = tmp_path / "comparison_output"

    config = ComparisonConfig(crop_size=64, zoom_factor=2)
    result = generate_comparison(
        quality_json_path=quality_path,
        output_dir=output_dir,
        project_root=project_root,
        config=config,
    )

    assert result.study_id == "test-comparison"
    assert len(result.strategies) >= 1
    for sr in result.strategies:
        assert sr.region.width == 64
        assert sr.output_dir.exists()
        assert len(sr.output_images) > 0
        for img_path in sr.output_images:
            assert img_path.exists()

        # Each strategy subdir has its own distortion map and annotated original
        assert (sr.output_dir / f"distortion_map_{sr.strategy}.webp").exists()
        assert (sr.output_dir / "original_annotated.webp").exists()
        assert (sr.output_dir / "distortion_map_comparison.webp").exists()
    # Intermediate files (encoded, crops) are in a temp dir and cleaned up
    assert not (output_dir / "encoded").exists()
    assert not (output_dir / "crops").exists()


# ---------------------------------------------------------------------------
# Tests: encode_image
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not shutil.which("cjpeg"),
    reason="cjpeg not available",
)
def test_encode_image_jpeg(sample_rgb_image: Path, tmp_output: Path) -> None:
    """Test re-encoding an image as JPEG."""
    measurement = {"format": "jpeg", "quality": 75}
    result = encode_image(sample_rgb_image, measurement, tmp_output)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".jpg"


@pytest.mark.skipif(
    not shutil.which("cwebp"),
    reason="cwebp not available",
)
def test_encode_image_webp(sample_rgb_image: Path, tmp_output: Path) -> None:
    """Test re-encoding an image as WebP."""
    measurement = {"format": "webp", "quality": 80, "method": 4}
    result = encode_image(sample_rgb_image, measurement, tmp_output)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".webp"


@pytest.mark.skipif(
    not shutil.which("avifenc"),
    reason="avifenc not available",
)
def test_encode_image_avif(sample_rgb_image: Path, tmp_output: Path) -> None:
    """Test re-encoding an image as AVIF."""
    measurement = {"format": "avif", "quality": 60, "speed": 6, "chroma_subsampling": "420"}
    result = encode_image(sample_rgb_image, measurement, tmp_output)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".avif"


@pytest.mark.skipif(
    not shutil.which("cjxl"),
    reason="cjxl not available",
)
def test_encode_image_jxl(sample_rgb_image: Path, tmp_output: Path) -> None:
    """Test re-encoding an image as JPEG XL."""
    measurement = {"format": "jxl", "quality": 70, "effort": 3}
    result = encode_image(sample_rgb_image, measurement, tmp_output)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".jxl"


def test_encode_image_unknown_format(sample_rgb_image: Path, tmp_output: Path) -> None:
    """Test that unknown format returns None."""
    measurement = {"format": "unknown", "quality": 50}
    result = encode_image(sample_rgb_image, measurement, tmp_output)
    assert result is None


# ---------------------------------------------------------------------------
# _resolve_encoded_path
# ---------------------------------------------------------------------------


def test_resolve_encoded_path_found(tmp_path: Path) -> None:
    """Returns absolute path when encoded file exists on disk."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    encoded = project_root / "data" / "encoded" / "study" / "jpeg" / "original"
    encoded.mkdir(parents=True)
    artifact = encoded / "img_q50.jpg"
    artifact.write_bytes(b"fake jpeg")

    measurement = {"encoded_path": "data/encoded/study/jpeg/original/img_q50.jpg"}
    result = _resolve_encoded_path(measurement, project_root)
    assert result is not None
    assert result == artifact


def test_resolve_encoded_path_missing(tmp_path: Path) -> None:
    """Returns None when encoded file does not exist."""
    measurement = {"encoded_path": "data/encoded/study/jpeg/original/img_q50.jpg"}
    result = _resolve_encoded_path(measurement, tmp_path)
    assert result is None


def test_resolve_encoded_path_empty() -> None:
    """Returns None when encoded_path is empty string."""
    measurement = {"encoded_path": ""}
    result = _resolve_encoded_path(measurement, Path("/tmp"))
    assert result is None


def test_resolve_encoded_path_not_present() -> None:
    """Returns None when encoded_path key is absent."""
    measurement: dict = {"format": "jpeg", "quality": 50}
    result = _resolve_encoded_path(measurement, Path("/tmp"))
    assert result is None


# ---------------------------------------------------------------------------
# _get_or_encode
# ---------------------------------------------------------------------------


def test_get_or_encode_uses_existing(tmp_path: Path) -> None:
    """Prefers pre-existing encoded file over re-encoding."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    artifact = project_root / "data" / "encoded" / "img.jpg"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"\xff\xd8fake")

    measurement = {"encoded_path": "data/encoded/img.jpg", "format": "jpeg", "quality": 50}
    # source_path doesn't need to exist since we should use the saved artifact
    source = tmp_path / "nonexistent_source.png"

    result = _get_or_encode(source, measurement, tmp_path / "out", project_root)
    assert result == artifact


@pytest.mark.skipif(
    not shutil.which("cjpeg"),
    reason="cjpeg not available",
)
def test_get_or_encode_falls_back_to_encoding(sample_rgb_image: Path, tmp_output: Path) -> None:
    """Falls back to re-encoding when no saved artifact exists."""
    measurement = {"encoded_path": "", "format": "jpeg", "quality": 75}
    project_root = tmp_output / "project"
    project_root.mkdir()

    result = _get_or_encode(sample_rgb_image, measurement, tmp_output, project_root)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".jpg"
