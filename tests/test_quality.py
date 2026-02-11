"""Tests for quality measurement module."""

from pathlib import Path

import pytest

from src.quality import EncodingResults, QualityMetrics


def test_quality_metrics_dataclass() -> None:
    """Test QualityMetrics dataclass."""
    metrics = QualityMetrics(ssimulacra2=85.5, psnr=42.3, ssim=0.98)
    assert metrics.ssimulacra2 == 85.5
    assert metrics.psnr == 42.3
    assert metrics.ssim == 0.98
    assert metrics.butteraugli is None


def test_encoding_results_from_file() -> None:
    """Test loading encoding results from JSON using committed fixture."""
    project_root = Path(__file__).parent.parent
    results_file = project_root / "tests" / "fixtures" / "encoding-results-fixture.json"

    # This fixture is committed to the repository, so it should always exist
    assert results_file.exists(), f"Fixture file not found: {results_file}"

    results = EncodingResults.from_file(results_file)

    assert results.study_id == "test-fixture-study"
    assert results.study_name == "Test Fixture Study"
    assert "id" in results.dataset
    assert results.dataset["id"] == "test-dataset"
    assert len(results.encodings) == 3  # JPEG, WebP, AVIF

    # Check first encoding record (JPEG)
    first = results.encodings[0]
    assert first.format == "jpeg"
    assert first.quality == 85
    assert first.file_size == 12345
    assert first.width == 100
    assert first.height == 100

    # Check that AVIF has extra parameters
    avif = results.encodings[2]
    assert avif.format == "avif"
    assert avif.chroma_subsampling == "420"
    assert avif.speed == 6


def test_encoding_results_file_not_found() -> None:
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        EncodingResults.from_file(Path("nonexistent.json"))
