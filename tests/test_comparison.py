"""Tests for visual comparison module."""

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.comparison import (
    ComparisonConfig,
    ComparisonResult,
    TargetComparisonResult,
    WorstRegion,
    _analysis_fragment_in_original,
    _anisotropic_std_map,
    _build_label,
    _build_metric_label,
    _default_tile_parameter,
    _format_figure_title,
    _render_distmap_thumbnail,
    _resolve_source_for_crop,
    assemble_comparison_grid,
    crop_and_zoom,
    determine_varying_parameters,
    encode_image,
    find_worst_region,
    generate_comparison,
    generate_distortion_map,
    load_quality_results,
    sort_tile_values,
)
from src.quality import find_worst_region_in_array, read_pfm

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
    """find_worst_region uses _read_pfm (now read_pfm) branch for .pfm files."""
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
# Tests: read_pfm
# ---------------------------------------------------------------------------


def test_read_pfm_grayscale(tmp_path: Path) -> None:
    """read_pfm returns correct values for a Pf (grayscale) PFM file."""
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    pfm_path = tmp_path / "gray.pfm"
    _write_pfm_grayscale(pfm_path, arr)

    result = read_pfm(pfm_path)
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, arr, rtol=1e-5)


def test_read_pfm_color(tmp_path: Path) -> None:
    """read_pfm returns max-across-channels for PF (colour) PFM files."""
    # Shape (2, 2, 3)
    arr = np.array(
        [[[1.0, 2.0, 0.5], [0.1, 0.2, 0.3]], [[3.0, 1.0, 2.0], [0.0, 0.0, 4.0]]], dtype=np.float32
    )
    pfm_path = tmp_path / "color.pfm"
    _write_pfm_color(pfm_path, arr)

    result = read_pfm(pfm_path)
    assert result.shape == (2, 2)
    # max across channels per pixel
    expected = np.array([[2.0, 0.3], [3.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_read_pfm_bottom_to_top_flip(tmp_path: Path) -> None:
    """read_pfm flips rows so the first row in result is the top of the image."""
    # Row 0 has value 10, row 1 has value 20
    arr = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    pfm_path = tmp_path / "flip.pfm"
    _write_pfm_grayscale(pfm_path, arr)

    result = read_pfm(pfm_path)
    # After flip, top row should still be [10, 10]
    np.testing.assert_allclose(result[0], [10.0, 10.0], rtol=1e-5)
    np.testing.assert_allclose(result[1], [20.0, 20.0], rtol=1e-5)


def test_read_pfm_invalid_magic(tmp_path: Path) -> None:
    """read_pfm raises ValueError for non-PFM files."""
    bad = tmp_path / "bad.pfm"
    bad.write_bytes(b"P6\n2 2\n255\n" + bytes(12))
    with pytest.raises(ValueError, match="Not a PFM file"):
        read_pfm(bad)


# ---------------------------------------------------------------------------
# Tests: find_worst_region_in_array
# ---------------------------------------------------------------------------


def test_find_worst_region_in_array_finds_hotspot() -> None:
    """find_worst_region_in_array locates the hot-spot in a float array."""
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:30, 40:50] = 10.0  # hot-spot at col 40-50, row 20-30
    region = find_worst_region_in_array(arr, crop_size=16)
    assert 24 <= region.x <= 50
    assert 4 <= region.y <= 30
    assert region.width == 16
    assert region.height == 16
    assert region.avg_distortion > 0


def test_find_worst_region_in_array_small_image() -> None:
    """Returns whole image when smaller than crop_size."""
    arr = np.full((20, 30), 5.0, dtype=np.float64)
    region = find_worst_region_in_array(arr, crop_size=64)
    assert region.x == 0
    assert region.y == 0
    assert region.width == 30
    assert region.height == 20
    assert abs(region.avg_distortion - 5.0) < 1e-9


def test_find_worst_region_in_array_uniform() -> None:
    """Any position is equally valid for a uniform array."""
    arr = np.full((128, 128), 3.0, dtype=np.float64)
    region = find_worst_region_in_array(arr, crop_size=32)
    assert region.width == 32
    assert region.height == 32
    assert abs(region.avg_distortion - 3.0) < 1e-9


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


@pytest.mark.skipif(
    not shutil.which("montage"),
    reason="ImageMagick montage not available",
)
def test_assemble_comparison_grid_figure_title(
    crop_images: list[tuple[Path, str, str]],
    tmp_output: Path,
) -> None:
    """Grid with figure_title should be taller than one without."""
    without_title = tmp_output / "grid_no_title.png"
    with_title = tmp_output / "grid_with_title.png"

    assemble_comparison_grid(crop_images, without_title, max_columns=4)
    assemble_comparison_grid(
        crop_images,
        with_title,
        max_columns=4,
        figure_title="Target: SSIMULACRA2 = 75",
    )

    assert with_title.exists()
    with Image.open(without_title) as no_t, Image.open(with_title) as with_t:
        assert with_t.height > no_t.height
        assert with_t.width == no_t.width


@pytest.mark.skipif(
    not shutil.which("montage"),
    reason="ImageMagick montage not available",
)
def test_assemble_comparison_grid_placeholder_indices(
    tmp_path: Path,
    tmp_output: Path,
) -> None:
    """Placeholder tiles should occupy the same space as normal tiles."""
    normal_crops = []
    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(i * 60, 100, 100))
        path = tmp_path / f"normal_{i}.png"
        img.save(path)
        normal_crops.append((path, f"Format {i}", f"Score: {i}"))

    placeholder_path = tmp_path / "placeholder.png"
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(placeholder_path)
    crops_with_placeholder = list(normal_crops)
    crops_with_placeholder[2] = (placeholder_path, "Skipped", "")

    out_full = tmp_output / "grid_full.png"
    out_placeholder = tmp_output / "grid_placeholder.png"

    assemble_comparison_grid(normal_crops, out_full, max_columns=4)
    assemble_comparison_grid(
        crops_with_placeholder,
        out_placeholder,
        max_columns=4,
        placeholder_indices=frozenset({2}),
    )

    assert out_placeholder.exists()
    with Image.open(out_full) as full, Image.open(out_placeholder) as ph:
        # Grids with same number of tiles should have the same dimensions
        assert full.width == ph.width
        assert full.height == ph.height


# ---------------------------------------------------------------------------
# Tests: _format_figure_title
# ---------------------------------------------------------------------------


def test_format_figure_title_ssimulacra2() -> None:
    assert _format_figure_title("ssimulacra2", 75) == "Target: SSIMULACRA2 = 75"


def test_format_figure_title_bytes_per_pixel() -> None:
    assert _format_figure_title("bytes_per_pixel", 0.1) == "Target: Bytes per pixel = 0.1"


def test_format_figure_title_psnr() -> None:
    assert _format_figure_title("psnr", 40.0) == "Target: PSNR = 40"


def test_format_figure_title_unknown_metric() -> None:
    title = _format_figure_title("some_custom_metric", 5)
    assert "5" in title
    assert "Some Custom Metric" in title


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
    arr = read_pfm(output_pfm)
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
    assert config.max_columns == 4
    assert config.label_font_size == 22
    assert config.distmap_vmax == 5.0
    assert config.source_image is None
    assert config.tile_parameter is None
    assert config.study_config_path is None


def test_comparison_config_custom() -> None:
    """Test custom configuration values."""
    config = ComparisonConfig(crop_size=96, zoom_factor=3)
    assert config.crop_size == 96
    assert config.zoom_factor == 3


def test_comparison_config_source_image_override() -> None:
    """Test source_image override."""
    config = ComparisonConfig(source_image="data/datasets/test/image1.png")
    assert config.source_image == "data/datasets/test/image1.png"


def test_comparison_result() -> None:
    """Test ComparisonResult dataclass."""
    region = WorstRegion(x=10, y=20, width=128, height=128, avg_distortion=150.0)
    tr = TargetComparisonResult(
        target_metric="ssimulacra2",
        target_value=70.0,
        source_image="image.png",
        region=region,
        interpolated_qualities={"jpeg": 65, "avif": 45},
        output_images=[Path("comparison.webp")],
    )
    result = ComparisonResult(
        study_id="test",
        target_results=[tr],
        varying_parameters=["format", "quality"],
    )
    assert result.study_id == "test"
    assert len(result.target_results) == 1
    assert result.target_results[0].region.x == 10
    assert result.target_results[0].target_metric == "ssimulacra2"
    assert result.target_results[0].target_value == 70.0


def test_target_comparison_result() -> None:
    """TargetComparisonResult records all per-target fields."""
    region = WorstRegion(x=0, y=0, width=64, height=64, avg_distortion=1.0)
    tr = TargetComparisonResult(
        target_metric="bytes_per_pixel",
        target_value=0.3,
        source_image="img.png",
        region=region,
        interpolated_qualities={"avif": 55.2},
    )
    assert tr.target_metric == "bytes_per_pixel"
    assert tr.target_value == 0.3
    assert tr.interpolated_qualities["avif"] == 55.2
    assert tr.output_images == []


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

    # Use only JPEG measurements
    jpeg_only_data = {
        **sample_quality_data,
        "measurements": [m for m in sample_quality_data["measurements"] if m["format"] == "jpeg"],
    }

    # Write quality.json
    quality_path = tmp_path / "quality.json"
    with open(quality_path, "w") as f:
        json.dump(jpeg_only_data, f)

    # Write study config with comparison targets
    study_config_path = tmp_path / "study_config.json"
    with open(study_config_path, "w") as f:
        json.dump(
            {
                "id": "test-comparison",
                "name": "Test Comparison Study",
                "dataset": {"id": "test"},
                "encoders": [{"format": "jpeg", "quality": [50, 80]}],
                "comparison": {
                    "targets": [{"metric": "ssimulacra2", "values": [60]}],
                },
            },
            f,
        )

    output_dir = tmp_path / "comparison_output"

    config = ComparisonConfig(crop_size=64, zoom_factor=2, study_config_path=study_config_path)
    result = generate_comparison(
        quality_json_path=quality_path,
        output_dir=output_dir,
        project_root=project_root,
        config=config,
    )

    assert result.study_id == "test-comparison"
    assert len(result.target_results) >= 1
    for tr in result.target_results:
        assert tr.region.width == 64
        assert tr.target_metric == "ssimulacra2"
        assert tr.target_value == 60
        assert len(tr.output_images) > 0
        for img_path in tr.output_images:
            assert img_path.exists()

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


@pytest.mark.skipif(
    not shutil.which("cjpeg"),
    reason="cjpeg not available",
)
# ---------------------------------------------------------------------------
# _default_tile_parameter
# ---------------------------------------------------------------------------


class TestDefaultTileParameter:
    """Tests for the _default_tile_parameter heuristic."""

    def test_format_takes_priority(self) -> None:
        """When 'format' is in varying, it is always chosen as tile param."""
        assert _default_tile_parameter(["quality", "format"]) == "format"
        assert _default_tile_parameter(["format", "speed"]) == "format"

    def test_non_quality_preferred_over_quality(self) -> None:
        """When format absent, first non-quality varying param is returned."""
        assert _default_tile_parameter(["quality", "speed"]) == "speed"
        assert _default_tile_parameter(["effort", "quality"]) == "effort"
        assert _default_tile_parameter(["quality", "chroma_subsampling"]) == "chroma_subsampling"

    def test_quality_only(self) -> None:
        """When only quality varies, quality is returned."""
        assert _default_tile_parameter(["quality"]) == "quality"

    def test_empty_list_returns_none(self) -> None:
        """Empty varying list returns None."""
        assert _default_tile_parameter([]) is None

    def test_method_as_non_quality(self) -> None:
        """Method is preferred over quality."""
        assert _default_tile_parameter(["method", "quality"]) == "method"

    def test_resolution_as_non_quality(self) -> None:
        """Resolution is preferred over quality."""
        assert _default_tile_parameter(["resolution", "quality"]) == "resolution"


# ---------------------------------------------------------------------------
# _anisotropic_std_map
# ---------------------------------------------------------------------------


class TestAnisotropicStdMap:
    """Tests for the fragment-selection std-dev distortion map."""

    def _arr(self, values: list[list[float]]) -> np.ndarray:
        return np.array(values, dtype=np.float64)

    def _pair(self, values: list[list[float]], **kwargs) -> tuple[np.ndarray, dict]:
        return self._arr(values), {"format": "jpeg", **kwargs}

    def test_single_variant_returns_map_itself(self) -> None:
        arr = self._arr([[1.0, 2.0], [3.0, 4.0]])
        result = _anisotropic_std_map([(arr, {})], [])
        np.testing.assert_array_almost_equal(result, arr)

    def test_two_variants_returns_pixelwise_std(self) -> None:
        a = self._arr([[1.0, 0.0], [0.0, 1.0]])
        b = self._arr([[3.0, 0.0], [0.0, 3.0]])
        result = _anisotropic_std_map([(a, {}), (b, {})], [])
        # std of [1,3] = 1.0; std of [0,0] = 0.0
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_std_not_normalised_by_mean(self) -> None:
        """Fragment selection uses absolute std, not CV: two high-distortion
        variants with the same *relative* spread should score higher than two
        low-distortion variants with the same relative spread."""
        high = [self._pair([[10.0, 10.0], [10.0, 10.0]]), self._pair([[30.0, 30.0], [30.0, 30.0]])]
        low = [self._pair([[1.0, 1.0], [1.0, 1.0]]), self._pair([[3.0, 3.0], [3.0, 3.0]])]
        std_high = _anisotropic_std_map(high, []).mean()
        std_low = _anisotropic_std_map(low, []).mean()
        assert std_high > std_low

    def test_groups_by_split_params(self) -> None:
        """Variants are grouped by split_params; each group's std is averaged."""
        # Two groups (value of 'quality'), each with 2 variants
        # Group q=50: [2,2] and [4,4] → std=1
        # Group q=80: [1,1] and [5,5] → std=2
        # Expected average std map = [[1.5, 1.5], [1.5, 1.5]]
        pairs = [
            (self._arr([[2.0, 2.0], [2.0, 2.0]]), {"quality": 50}),
            (self._arr([[4.0, 4.0], [4.0, 4.0]]), {"quality": 50}),
            (self._arr([[1.0, 1.0], [1.0, 1.0]]), {"quality": 80}),
            (self._arr([[5.0, 5.0], [5.0, 5.0]]), {"quality": 80}),
        ]
        result = _anisotropic_std_map(pairs, ["quality"])
        np.testing.assert_array_almost_equal(result, np.full((2, 2), 1.5))

    def test_fallback_when_no_group_has_two_variants(self) -> None:
        """When each split-param group has only 1 variant, falls back to pooled std."""
        pairs = [
            (self._arr([[2.0, 2.0], [2.0, 2.0]]), {"quality": 50}),
            (self._arr([[4.0, 4.0], [4.0, 4.0]]), {"quality": 80}),
        ]
        result = _anisotropic_std_map(pairs, ["quality"])
        # Pooled std of [2,4] = 1.0
        np.testing.assert_array_almost_equal(result, np.full((2, 2), 1.0))


# ---------------------------------------------------------------------------
# Tests: sort_tile_values
# ---------------------------------------------------------------------------


class TestSortTileValues:
    """Tests for the numeric tile-value sorting helper."""

    def test_numeric_effort_values_sorted_numerically(self) -> None:
        """Effort values 1-10 must come out in numeric order, not lexicographic."""
        raw = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
        result = sort_tile_values(raw)
        assert result == ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    def test_ten_after_nine_not_after_one(self) -> None:
        """The key regression: '10' must come after '9', not after '1'."""
        result = sort_tile_values({"1", "10", "2"})
        assert result == ["1", "2", "10"]

    def test_format_strings_sorted_lexicographically(self) -> None:
        """Non-numeric values fall back to lexicographic sort."""
        result = sort_tile_values({"webp", "avif", "jpeg", "jxl"})
        assert result == sorted(["webp", "avif", "jpeg", "jxl"])

    def test_single_value(self) -> None:
        """Single-element input is returned as a one-element list."""
        assert sort_tile_values({"5"}) == ["5"]

    def test_empty(self) -> None:
        """Empty input returns empty list."""
        assert sort_tile_values(set()) == []

    def test_float_values_sorted_numerically(self) -> None:
        """Float-representation strings (e.g. quality 0.1 / 0.5 / 1.0) sort numerically."""
        result = sort_tile_values({"0.5", "0.1", "1.0"})
        assert result == ["0.1", "0.5", "1.0"]

    def test_accepts_list(self) -> None:
        """Accepts a plain list as well as a set."""
        result = sort_tile_values(["10", "1", "2"])
        assert result == ["1", "2", "10"]


# ---------------------------------------------------------------------------
# Tests: per-tile interpolation for non-format tile_param
# ---------------------------------------------------------------------------


def _make_jxl_effort_measurements(image: str = "img.png") -> list[dict]:
    """Create synthetic JXL effort-sweep measurements.

    Effort 1 (fast/low quality): ssimulacra2 climbs steeply with quality.
    Effort 7 (slow/high quality): ssimulacra2 is notably higher at the same
    encoder quality setting, simulating the real-world gain from higher effort.
    This ensures that interpolating to the same ssimulacra2 target gives a
    *different* encoder quality for each effort level.
    """
    base = {
        "format": "jxl",
        "width": 100,
        "height": 100,
        "measurement_error": None,
        "speed": None,
        "method": None,
        "chroma_subsampling": None,
        "resolution": None,
        "source_image": image,
        "original_image": image,
    }
    measurements = []
    for quality in (40, 50, 60, 70, 80):
        # effort=1: each quality point adds 1 unit of ssimulacra2 starting at 40
        ssim1 = 40.0 + (quality - 40) * 1.0
        measurements.append(
            {
                **base,
                "quality": quality,
                "effort": 1,
                "ssimulacra2": ssim1,
                "file_size": 1000 + quality * 10,
            }
        )
        # effort=7: 10 units better than effort=1 at the same encoder quality
        ssim7 = ssim1 + 10.0
        measurements.append(
            {
                **base,
                "quality": quality,
                "effort": 7,
                "ssimulacra2": ssim7,
                "file_size": 900 + quality * 10,  # slightly smaller at same quality
            }
        )
    return measurements


class TestEffortSweepInterpolation:
    """Tests covering the fixes for effort-sweep (tile_param != 'format') studies."""

    def test_interpolate_quality_differs_per_effort(self) -> None:
        """Each effort level must yield a different encoder quality for the same target.

        This is the core regression: before the fix, all effort levels shared the
        same interpolated quality because the effort filter was not passed.
        """
        from src.interpolation import interpolate_quality_for_metric

        ms = _make_jxl_effort_measurements()
        target_ssim = 60.0

        q_effort1 = interpolate_quality_for_metric(ms, "jxl", "ssimulacra2", target_ssim, effort=1)
        q_effort7 = interpolate_quality_for_metric(ms, "jxl", "ssimulacra2", target_ssim, effort=7)

        assert q_effort1 is not None
        assert q_effort7 is not None
        # effort=7 needs a *lower* encoder quality to hit the same ssimulacra2 target
        assert q_effort1 > q_effort7, (
            f"effort=1 quality ({q_effort1:.2f}) should be higher than "
            f"effort=7 quality ({q_effort7:.2f}) to achieve the same ssimulacra2"
        )

    def test_interpolate_quality_without_effort_gives_averaged_result(self) -> None:
        """Without an effort filter the result is an average-case estimate.

        When effort is not specified, the interpolation averages across all
        effort levels.  The averaged quality should fall between the per-effort
        values (not be identical to either), confirming that effort= filtering
        is necessary for per-tile accuracy.
        """
        from src.interpolation import interpolate_quality_for_metric

        ms = _make_jxl_effort_measurements()
        target_ssim = 60.0

        q_effort1 = interpolate_quality_for_metric(ms, "jxl", "ssimulacra2", target_ssim, effort=1)
        q_effort7 = interpolate_quality_for_metric(ms, "jxl", "ssimulacra2", target_ssim, effort=7)
        q_no_filter = interpolate_quality_for_metric(ms, "jxl", "ssimulacra2", target_ssim)

        assert q_effort1 is not None
        assert q_effort7 is not None
        assert q_no_filter is not None
        lo, hi = min(q_effort1, q_effort7), max(q_effort1, q_effort7)
        assert lo <= q_no_filter <= hi, "No-effort result should be between the per-effort values"

    def test_interpolation_at_exact_quality_is_circular(self) -> None:
        """Interpolating a metric at the exact float quality that was chosen to hit
        a target reproduces that target exactly — a circular result.

        This test documents *why* interpolation must not be used for metric labels
        in comparison figures.  The comparison script instead measures metrics
        directly from the encoded file using QualityMeasurer, which produces the
        true (non-circular) measured value.
        """
        from src.interpolation import (
            interpolate_metric_at_quality,
            interpolate_quality_for_metric,
        )

        ms = _make_jxl_effort_measurements()
        target_ssim = 62.7  # not an exact data point

        quality = interpolate_quality_for_metric(ms, "jxl", "ssimulacra2", target_ssim, effort=1)
        assert quality is not None

        # Evaluating the metric at the float quality that was designed to hit
        # target_ssim will return target_ssim (circular by construction).
        val = interpolate_metric_at_quality(ms, "jxl", quality, "ssimulacra2", effort=1)
        assert val is not None
        assert val == pytest.approx(target_ssim, abs=0.01), (
            "Interpolating at the exact float quality must reproduce the target "
            "(documenting the circularity that makes interpolation unsuitable for labels)"
        )

    def test_interpolated_qualities_keyed_by_tile_value(self) -> None:
        """interpolated_qualities must use the tile value as key, not the format.

        When tile_param='effort', every effort level has format='jxl'.
        If the key were fmt ('jxl'), only the last effort's quality would
        survive in the dict.  The key must be the tile value (e.g. '1', '7')
        so that every effort level retains its own interpolated quality setting.
        """
        from src.interpolation import interpolate_quality_for_metric

        ms = _make_jxl_effort_measurements()
        target_ssim = 60.0

        tile_values = sort_tile_values({"1", "7"})
        interpolated_qualities: dict[str, float] = {}
        for tv in tile_values:
            example = next(
                m for m in ms if str(m.get("effort")) == tv and m.get("measurement_error") is None
            )
            q = interpolate_quality_for_metric(
                ms, "jxl", "ssimulacra2", target_ssim, effort=example["effort"]
            )
            assert q is not None
            # Simulate the fixed code: key by tv, not by fmt
            interpolated_qualities[tv] = q

        assert "1" in interpolated_qualities
        assert "7" in interpolated_qualities
        assert interpolated_qualities["1"] != interpolated_qualities["7"], (
            "Different efforts must produce different interpolated qualities"
        )


# ---------------------------------------------------------------------------
# _resolve_source_for_crop tests
# ---------------------------------------------------------------------------


class TestResolveSourceForCrop:
    """Tests for _resolve_source_for_crop image-filtering logic."""

    def _make_crop_measurement(
        self,
        *,
        original_image: str,
        crop: int,
        crop_region: dict[str, int],
        analysis_fragment: dict[str, int] | None = None,
    ) -> dict:
        """Build a minimal measurement dict with crop_region info."""
        return {
            "source_image": original_image,
            "original_image": original_image,
            "format": "jpeg",
            "quality": 50,
            "file_size": 50000,
            "width": crop_region["width"],
            "height": crop_region["height"],
            "source_file_size": 200000,
            "ssimulacra2": 70.0,
            "butteraugli": 3.0,
            "crop": crop,
            "crop_region": crop_region,
            "analysis_fragment": analysis_fragment,
            "measurement_error": None,
        }

    def test_filters_by_selected_image(self, tmp_path: Path) -> None:
        """Must use crop_region from the selected image, not others."""
        from tests.conftest import create_test_image

        # Create two images with different aspect ratios
        landscape = create_test_image(
            tmp_path / "landscape.png", size=(2000, 1000)
        )
        portrait = create_test_image(
            tmp_path / "portrait.png", size=(1000, 2000)
        )

        # Measurements from both images at crop level 800
        measurements = [
            self._make_crop_measurement(
                original_image="data/datasets/test/landscape.png",
                crop=800,
                crop_region={"x": 100, "y": 0, "width": 800, "height": 400},
                analysis_fragment={"x": 200, "y": 100, "width": 200, "height": 200},
            ),
            self._make_crop_measurement(
                original_image="data/datasets/test/portrait.png",
                crop=800,
                crop_region={"x": 0, "y": 100, "width": 400, "height": 800},
                analysis_fragment={"x": 100, "y": 200, "width": 200, "height": 200},
            ),
        ]

        cache: dict[int, tuple[Path, dict, dict | None]] = {}

        # Select portrait image — should use portrait's crop_region
        _, crop_region, _ = _resolve_source_for_crop(
            portrait,
            800,
            measurements,
            cache,
            tmp_path / "work",
            selected_image="data/datasets/test/portrait.png",
        )

        assert crop_region is not None
        # Portrait crop: width < height (400x800)
        assert crop_region["width"] == 400
        assert crop_region["height"] == 800

    def test_wrong_image_without_filter(self, tmp_path: Path) -> None:
        """Without selected_image filter, may pick wrong image's region."""
        from tests.conftest import create_test_image

        portrait = create_test_image(
            tmp_path / "portrait.png", size=(1000, 2000)
        )

        # Landscape measurement listed first
        measurements = [
            self._make_crop_measurement(
                original_image="data/datasets/test/landscape.png",
                crop=800,
                crop_region={"x": 100, "y": 0, "width": 800, "height": 400},
            ),
            self._make_crop_measurement(
                original_image="data/datasets/test/portrait.png",
                crop=800,
                crop_region={"x": 0, "y": 100, "width": 400, "height": 800},
            ),
        ]

        cache: dict[int, tuple[Path, dict, dict | None]] = {}

        # Without filter — picks first match (landscape region, wrong for portrait)
        _, crop_region, _ = _resolve_source_for_crop(
            portrait,
            800,
            measurements,
            cache,
            tmp_path / "work",
            # No selected_image
        )

        assert crop_region is not None
        # Gets landscape's region (width > height), wrong for portrait
        assert crop_region["width"] == 800
        assert crop_region["height"] == 400

    def test_fallback_when_no_stored_region(self, tmp_path: Path) -> None:
        """Falls back to preprocessing when no measurement matches."""
        from tests.conftest import create_test_image

        img = create_test_image(
            tmp_path / "img.png", size=(2000, 1000)
        )

        cache: dict[int, tuple[Path, dict, dict | None]] = {}

        # No measurements at all
        _, crop_region, _ = _resolve_source_for_crop(
            img,
            800,
            [],
            cache,
            tmp_path / "work",
            selected_image="data/datasets/test/img.png",
        )

        assert crop_region is not None
        # 2000x1000, longest=2000, scale=800/2000=0.4
        # crop_w = round(2000*0.4) = 800, crop_h = round(1000*0.4) = 400
        assert crop_region["width"] == 800
        assert crop_region["height"] == 400

    def test_crop_preserves_aspect_ratio(self, tmp_path: Path) -> None:
        """Crop region from stored data must preserve aspect ratio."""
        from tests.conftest import create_test_image

        img = create_test_image(
            tmp_path / "img.png", size=(2040, 1356)
        )

        # Simulate pipeline-produced measurements for multiple crop levels
        orig_w, orig_h = 2040, 1356
        orig_longest = 2040
        orig_ar = orig_w / orig_h
        fragment = {"x": 500, "y": 400, "width": 200, "height": 200}

        measurements = []
        for crop_level in [1600, 1200, 800, 400]:
            scale = crop_level / orig_longest
            cw = max(1, round(orig_w * scale))
            ch = max(1, round(orig_h * scale))
            # Center fragment
            frag_cx = fragment["x"] + fragment["width"] / 2
            frag_cy = fragment["y"] + fragment["height"] / 2
            cx = max(0, min(int(round(frag_cx - cw / 2)), orig_w - cw))
            cy = max(0, min(int(round(frag_cy - ch / 2)), orig_h - ch))
            measurements.append(
                self._make_crop_measurement(
                    original_image="data/test/img.png",
                    crop=crop_level,
                    crop_region={"x": cx, "y": cy, "width": cw, "height": ch},
                    analysis_fragment={
                        "x": fragment["x"] - cx,
                        "y": fragment["y"] - cy,
                        "width": 200,
                        "height": 200,
                    },
                )
            )

        for crop_level in [1600, 1200, 800, 400]:
            cache: dict[int, tuple[Path, dict, dict | None]] = {}
            _, crop_region, _ = _resolve_source_for_crop(
                img,
                crop_level,
                measurements,
                cache,
                tmp_path / f"work_{crop_level}",
                selected_image="data/test/img.png",
            )
            assert crop_region is not None
            crop_ar = crop_region["width"] / crop_region["height"]
            assert abs(crop_ar - orig_ar) < 0.01, (
                f"Crop level {crop_level}: aspect ratio {crop_ar:.4f} "
                f"doesn't match original {orig_ar:.4f}"
            )


# ---------------------------------------------------------------------------
# _analysis_fragment_in_original tests
# ---------------------------------------------------------------------------


class TestAnalysisFragmentInOriginal:
    """Tests for _analysis_fragment_in_original image-filtering logic."""

    def test_uses_crop_cache_first(self) -> None:
        """Should use crop_cache before falling back to measurements."""
        crop_cache: dict[int, tuple[Path, dict, dict | None]] = {
            800: (
                Path("/tmp/crop.png"),
                {"x": 100, "y": 50, "width": 800, "height": 400},
                {"x": 200, "y": 150, "width": 200, "height": 200},
            )
        }
        result = _analysis_fragment_in_original(
            crop_cache,
            [],
            selected_image="data/test/img.png",
        )
        assert result is not None
        # x = crop_x + frag_x = 100 + 200 = 300
        assert result["x"] == 300
        # y = crop_y + frag_y = 50 + 150 = 200
        assert result["y"] == 200
        assert result["width"] == 200
        assert result["height"] == 200

    def test_fallback_filters_by_selected_image(self) -> None:
        """Fallback scan must only use measurements for the selected image."""
        measurements = [
            {
                "original_image": "data/test/other.png",
                "crop": 800,
                "crop_region": {"x": 999, "y": 999, "width": 100, "height": 100},
                "analysis_fragment": {"x": 10, "y": 10, "width": 200, "height": 200},
            },
            {
                "original_image": "data/test/selected.png",
                "crop": 800,
                "crop_region": {"x": 50, "y": 30, "width": 800, "height": 400},
                "analysis_fragment": {"x": 100, "y": 70, "width": 200, "height": 200},
            },
        ]

        # Empty crop_cache forces fallback
        result = _analysis_fragment_in_original(
            {},
            measurements,
            selected_image="data/test/selected.png",
        )
        assert result is not None
        # Should use selected.png's data: x=50+100=150, y=30+70=100
        assert result["x"] == 150
        assert result["y"] == 100

    def test_fallback_without_filter_picks_first(self) -> None:
        """Without selected_image, fallback picks first match."""
        measurements = [
            {
                "original_image": "data/test/other.png",
                "crop_region": {"x": 999, "y": 999, "width": 100, "height": 100},
                "analysis_fragment": {"x": 10, "y": 10, "width": 200, "height": 200},
            },
            {
                "original_image": "data/test/selected.png",
                "crop_region": {"x": 50, "y": 30, "width": 800, "height": 400},
                "analysis_fragment": {"x": 100, "y": 70, "width": 200, "height": 200},
            },
        ]

        result = _analysis_fragment_in_original(
            {},
            measurements,
            # No selected_image — picks first
        )
        assert result is not None
        # Gets other.png's data: x=999+10=1009
        assert result["x"] == 1009


# ---------------------------------------------------------------------------
# Preprocessing aspect-ratio tests (crop_image_around_fragment)
# ---------------------------------------------------------------------------


class TestCropAspectRatioPreservation:
    """Verify that crop_image_around_fragment preserves the original AR."""

    @pytest.mark.parametrize(
        "orig_size,crop_levels",
        [
            ((2040, 1356), [1600, 1200, 800, 400]),
            ((1356, 2040), [1600, 1200, 800, 400]),
            ((2000, 2000), [1600, 1200, 800, 400]),
            ((3000, 2000), [2048, 1500, 1000, 500]),
            ((1920, 1080), [1600, 1200, 800, 400]),
        ],
    )
    def test_aspect_ratio_preserved_for_various_sizes(
        self,
        tmp_path: Path,
        orig_size: tuple[int, int],
        crop_levels: list[int],
    ) -> None:
        """Aspect ratio is preserved within rounding tolerance."""
        from tests.conftest import create_test_image
        from src.preprocessing import ImagePreprocessor

        img_path = create_test_image(
            tmp_path / "src.png", size=orig_size
        )
        orig_w, orig_h = orig_size
        orig_longest = max(orig_w, orig_h)
        orig_ar = orig_w / orig_h

        # Place fragment at center
        fragment = {
            "x": orig_w // 2 - 100,
            "y": orig_h // 2 - 100,
            "width": 200,
            "height": 200,
        }

        for crop_level in crop_levels:
            if crop_level >= orig_longest:
                continue
            # Check if fragment fits
            scale = crop_level / orig_longest
            cw = max(1, round(orig_w * scale))
            ch = max(1, round(orig_h * scale))
            if fragment["width"] > cw or fragment["height"] > ch:
                continue

            preprocessor = ImagePreprocessor(
                tmp_path / f"out_{crop_level}"
            )
            result = preprocessor.crop_image_around_fragment(
                img_path,
                fragment=fragment,
                target_longest_edge=crop_level,
            )

            cr = result.crop_region
            crop_ar = cr["width"] / cr["height"]
            assert abs(crop_ar - orig_ar) < 0.01, (
                f"Image {orig_w}x{orig_h}, crop {crop_level}: "
                f"AR {crop_ar:.4f} != original {orig_ar:.4f}"
            )

            # Verify crop stays within image bounds
            assert cr["x"] >= 0
            assert cr["y"] >= 0
            assert cr["x"] + cr["width"] <= orig_w
            assert cr["y"] + cr["height"] <= orig_h

            # Verify fragment is contained
            assert cr["x"] <= fragment["x"]
            assert cr["y"] <= fragment["y"]
            assert cr["x"] + cr["width"] >= fragment["x"] + fragment["width"]
            assert cr["y"] + cr["height"] >= fragment["y"] + fragment["height"]

    @pytest.mark.parametrize(
        "orig_size,frag_pos",
        [
            ((2000, 3000), (500, 500)),   # top-left region
            ((2000, 3000), (1700, 2700)), # bottom-right region
            ((2000, 3000), (0, 0)),       # top-left corner
            ((2000, 3000), (1800, 2800)), # near bottom-right corner
            ((3000, 2000), (100, 100)),   # landscape, top-left
            ((3000, 2000), (2700, 1700)), # landscape, bottom-right
        ],
    )
    def test_crop_stays_within_bounds_edge_fragments(
        self,
        tmp_path: Path,
        orig_size: tuple[int, int],
        frag_pos: tuple[int, int],
    ) -> None:
        """Crop regions never extend outside image boundaries."""
        from tests.conftest import create_test_image
        from src.preprocessing import ImagePreprocessor

        orig_w, orig_h = orig_size
        fx, fy = frag_pos
        frag_w = min(200, orig_w - fx)
        frag_h = min(200, orig_h - fy)
        fragment = {"x": fx, "y": fy, "width": frag_w, "height": frag_h}

        img_path = create_test_image(
            tmp_path / "src.png", size=orig_size
        )

        orig_longest = max(orig_w, orig_h)
        for crop_level in [1500, 1000, 500]:
            if crop_level >= orig_longest:
                continue
            scale = crop_level / orig_longest
            cw = max(1, round(orig_w * scale))
            ch = max(1, round(orig_h * scale))
            if frag_w > cw or frag_h > ch:
                continue

            preprocessor = ImagePreprocessor(
                tmp_path / f"out_{frag_pos}_{crop_level}"
            )
            result = preprocessor.crop_image_around_fragment(
                img_path,
                fragment=fragment,
                target_longest_edge=crop_level,
            )

            cr = result.crop_region
            assert cr["x"] >= 0, f"crop_x={cr['x']} < 0"
            assert cr["y"] >= 0, f"crop_y={cr['y']} < 0"
            assert cr["x"] + cr["width"] <= orig_w, (
                f"crop extends to x={cr['x'] + cr['width']} > image width {orig_w}"
            )
            assert cr["y"] + cr["height"] <= orig_h, (
                f"crop extends to y={cr['y'] + cr['height']} > image height {orig_h}"
            )

            # Verify actual image dimensions match crop_region
            with Image.open(result.path) as img:
                assert img.size == (cr["width"], cr["height"])
