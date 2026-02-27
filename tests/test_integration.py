"""Integration tests for encoding, quality measurement, and basic module smoke tests."""

from pathlib import Path

import pytest

from src.encoder import ImageEncoder
from src.quality import QualityMeasurer, QualityMetrics

# ---------------------------------------------------------------------------
# Basic module smoke tests (merged from test_encoder.py and test_quality.py)
# ---------------------------------------------------------------------------


def test_encoder_initialization(tmp_path: Path) -> None:
    """Test ImageEncoder initialization."""
    encoder = ImageEncoder(tmp_path / "output")
    assert encoder.output_dir.exists()


def test_quality_metrics_dataclass() -> None:
    """Test QualityMetrics dataclass."""
    metrics = QualityMetrics(ssimulacra2=85.5, psnr=42.3, ssim=0.98)
    assert metrics.ssimulacra2 == 85.5
    assert metrics.psnr == 42.3
    assert metrics.ssim == 0.98
    assert metrics.butteraugli is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_image() -> Path:
    """Return path to the test fixture image (PNG format)."""
    return Path(__file__).parent / "fixtures" / "test_image.png"


@pytest.fixture
def test_image_ppm() -> Path:
    """Return path to the test fixture image in PPM format for cjpeg/avifenc."""
    return Path(__file__).parent / "fixtures" / "test_image.ppm"


@pytest.fixture
def temp_results_dir(tmp_path) -> Path:
    """Create a temporary directory for test results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def encoder(temp_results_dir) -> ImageEncoder:
    """Create an ImageEncoder instance with temp output directory."""
    return ImageEncoder(temp_results_dir)


@pytest.fixture
def measurer() -> QualityMeasurer:
    """Create a QualityMeasurer instance."""
    return QualityMeasurer()


class TestEncodingPipeline:
    """Test the complete encoding pipeline for all formats."""

    def test_jpeg_encoding(self, encoder: ImageEncoder, test_image_ppm: Path):
        """Test JPEG encoding produces valid output."""
        result = encoder.encode_jpeg(test_image_ppm, quality=85)

        assert result.success, f"JPEG encoding failed: {result.error_message}"
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.file_size > 0

    def test_webp_encoding(self, encoder: ImageEncoder, test_image: Path):
        """Test WebP encoding produces valid output."""
        result = encoder.encode_webp(test_image, quality=85)

        assert result.success, f"WebP encoding failed: {result.error_message}"
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.file_size > 0

    def test_avif_encoding(self, encoder: ImageEncoder, test_image: Path):
        """Test AVIF encoding produces valid output.

        Note: Requires avifenc with codec support (e.g., libaom, librav1e).
        """
        result = encoder.encode_avif(test_image, quality=60, speed=6)

        assert result.success, f"AVIF encoding failed: {result.error_message}"
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.file_size > 0

    def test_jxl_encoding(self, encoder: ImageEncoder, test_image: Path):
        """Test JPEG XL encoding produces valid output."""
        result = encoder.encode_jxl(test_image, quality=85)

        assert result.success, f"JPEG XL encoding failed: {result.error_message}"
        assert result.output_path is not None
        assert result.output_path.exists()
        assert result.file_size > 0


class TestQualityMeasurementPipeline:
    """Test the complete quality measurement pipeline."""

    def test_quality_measurement_after_encoding(
        self,
        encoder: ImageEncoder,
        measurer: QualityMeasurer,
        test_image_ppm: Path,
        test_image: Path,
    ):
        """Test quality measurements work on encoded images."""
        # Encode with JPEG first (using PPM input)
        result = encoder.encode_jpeg(test_image_ppm, quality=85)
        assert result.success and result.output_path is not None

        # Measure quality against original PNG (quality tools work with PNG)
        metrics = measurer.measure_all(test_image, result.output_path)

        # Verify at least PSNR and SSIM are computed (always available via ffmpeg)
        assert metrics.psnr is not None
        assert metrics.ssim is not None

        # Sanity check on metric values
        assert 0 <= metrics.ssim <= 1  # SSIM is between 0 and 1
        assert metrics.psnr > 0  # PSNR should be positive

        # SSIMULACRA2 and Butteraugli may fail on very small images, but that's OK for smoke testing

    def test_quality_measurement_all_formats(
        self,
        encoder: ImageEncoder,
        measurer: QualityMeasurer,
        test_image: Path,
        test_image_ppm: Path,
    ):
        """Test quality measurements work for all encoding formats."""
        formats_to_test = [
            ("JPEG", lambda: encoder.encode_jpeg(test_image_ppm, quality=85)),
            ("WebP", lambda: encoder.encode_webp(test_image, quality=85)),
            ("AVIF", lambda: encoder.encode_avif(test_image, quality=60, speed=6)),
            ("JPEG XL", lambda: encoder.encode_jxl(test_image, quality=85)),
        ]

        for format_name, encode_func in formats_to_test:
            result = encode_func()
            assert result.success, (
                f"{format_name} encoding failed: {result.error_message if hasattr(result, 'error_message') else 'unknown error'}"
            )
            assert result.output_path is not None

            metrics = measurer.measure_all(test_image, result.output_path)

            # At least PSNR and SSIM should be available (via ffmpeg)
            assert metrics.psnr is not None, f"PSNR missing for {format_name}"
            assert metrics.ssim is not None, f"SSIM missing for {format_name}"
            # SSIMULACRA2 and Butteraugli may fail on very small test images


class TestEndToEndWorkflow:
    """Test the complete end-to-end workflow."""

    def test_complete_workflow(
        self, encoder: ImageEncoder, measurer: QualityMeasurer, test_image: Path
    ):
        """Test a complete encoding and measurement workflow."""
        # Test with multiple quality levels
        quality_levels = [60, 75, 90]

        for quality in quality_levels:
            # Encode
            result = encoder.encode_webp(test_image, quality=quality)
            assert result.success
            assert result.output_path is not None

            # Measure
            metrics = measurer.measure_all(test_image, result.output_path)

            # Verify we got valid measurements (at minimum PSNR and SSIM via ffmpeg)
            assert metrics.psnr is not None, f"Missing PSNR for quality {quality}"
            assert metrics.ssim is not None, f"Missing SSIM for quality {quality}"

            # Higher quality should generally give better scores
            # (though this is a very small test image, so results may vary)
            assert metrics.psnr > 0
            assert 0 <= metrics.ssim <= 1
