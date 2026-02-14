"""Tests for analysis module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import (
    METRIC_DIRECTIONS,
    analyze_study,
    compute_statistics,
    create_analysis_dataframe,
    determine_sweep_parameter,
    determine_varying_parameters,
    get_worst_percentile_col,
    load_quality_results,
)


@pytest.fixture
def sample_quality_data() -> dict:
    """Create sample quality measurement data for testing."""
    return {
        "study_id": "test-study",
        "study_name": "Test Study",
        "dataset": {
            "id": "test-dataset",
            "path": "data/datasets/test",
            "image_count": 2,
        },
        "encoding_timestamp": "2026-02-11T12:00:00+00:00",
        "timestamp": "2026-02-11T12:30:00+00:00",
        "measurements": [
            {
                "source_image": "data/datasets/test/image1.png",
                "original_image": "data/datasets/test/image1.png",
                "encoded_path": "data/encoded/test/avif/image1_q50.avif",
                "format": "avif",
                "quality": 50,
                "file_size": 100000,
                "width": 1920,
                "height": 1080,
                "source_file_size": 1000000,
                "ssimulacra2": 75.5,
                "psnr": 38.5,
                "ssim": 0.95,
                "butteraugli": 2.5,
                "chroma_subsampling": "420",
                "speed": 4,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image2.png",
                "original_image": "data/datasets/test/image2.png",
                "encoded_path": "data/encoded/test/avif/image2_q50.avif",
                "format": "avif",
                "quality": 50,
                "file_size": 120000,
                "width": 1920,
                "height": 1080,
                "source_file_size": 1100000,
                "ssimulacra2": 73.2,
                "psnr": 37.8,
                "ssim": 0.94,
                "butteraugli": 2.8,
                "chroma_subsampling": "420",
                "speed": 4,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image1.png",
                "original_image": "data/datasets/test/image1.png",
                "encoded_path": "data/encoded/test/avif/image1_q70.avif",
                "format": "avif",
                "quality": 70,
                "file_size": 150000,
                "width": 1920,
                "height": 1080,
                "source_file_size": 1000000,
                "ssimulacra2": 82.3,
                "psnr": 40.2,
                "ssim": 0.97,
                "butteraugli": 1.8,
                "chroma_subsampling": "420",
                "speed": 4,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "data/datasets/test/image2.png",
                "original_image": "data/datasets/test/image2.png",
                "encoded_path": "data/encoded/test/avif/image2_q70.avif",
                "format": "avif",
                "quality": 70,
                "file_size": 170000,
                "width": 1920,
                "height": 1080,
                "source_file_size": 1100000,
                "ssimulacra2": 80.1,
                "psnr": 39.5,
                "ssim": 0.96,
                "butteraugli": 2.0,
                "chroma_subsampling": "420",
                "speed": 4,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
        ],
    }


def test_load_quality_results(tmp_path: Path, sample_quality_data: dict) -> None:
    """Test loading quality results from JSON file."""
    quality_file = tmp_path / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(sample_quality_data, f)

    results = load_quality_results(quality_file)

    assert results["study_id"] == "test-study"
    assert len(results["measurements"]) == 4


def test_create_analysis_dataframe(sample_quality_data: dict) -> None:
    """Test creating analysis dataframe with derived metrics."""
    df = create_analysis_dataframe(sample_quality_data)

    assert len(df) == 4
    assert "bytes_per_pixel" in df.columns
    assert "compression_ratio" in df.columns
    assert "bytes_per_ssimulacra2_per_pixel" in df.columns

    # Check bytes_per_pixel calculation
    # Image is 1920 * 1080 = 2,073,600 pixels
    expected_bpp_first = 100000 / (1920 * 1080)
    assert abs(df.iloc[0]["bytes_per_pixel"] - expected_bpp_first) < 0.0001

    # Check compression ratio
    expected_ratio = 1000000 / 100000
    assert abs(df.iloc[0]["compression_ratio"] - expected_ratio) < 0.001


def test_compute_statistics(sample_quality_data: dict) -> None:
    """Test computing statistics grouped by parameters."""
    df = create_analysis_dataframe(sample_quality_data)
    stats = compute_statistics(df, ["format", "quality"])

    # Should have 2 rows (quality 50 and 70)
    assert len(stats) == 2

    # Check that statistical columns exist
    assert "ssimulacra2_mean" in stats.columns
    assert "ssimulacra2_p05" in stats.columns
    assert "ssimulacra2_p95" in stats.columns

    # Check mean for quality 50
    q50_stats = stats[stats["quality"] == 50].iloc[0]
    expected_mean = (75.5 + 73.2) / 2
    assert abs(q50_stats["ssimulacra2_mean"] - expected_mean) < 0.01


def test_determine_sweep_parameter(sample_quality_data: dict) -> None:
    """Test determining which parameter is swept."""
    df = create_analysis_dataframe(sample_quality_data)

    # Quality varies (50, 70), so it should be selected
    sweep_param = determine_sweep_parameter(df)
    assert sweep_param == "quality"


def test_determine_sweep_parameter_resolution() -> None:
    """Test sweep parameter detection with resolution."""
    df = pd.DataFrame(
        {
            "quality": [50, 50, 50, 50],
            "resolution": [640, 960, 1280, 1920],
            "format": ["avif", "avif", "avif", "avif"],
        }
    )

    sweep_param = determine_sweep_parameter(df)
    assert sweep_param == "resolution"


def test_determine_varying_parameters() -> None:
    """Test detection of varying parameters."""
    df = pd.DataFrame(
        {
            "quality": [50, 60, 70, 80],
            "resolution": [1920, 1920, 1920, 1920],
            "format": ["avif", "avif", "jpeg", "jpeg"],
            "speed": [4, 4, 4, 4],
        }
    )

    varying = determine_varying_parameters(df)
    assert "quality" in varying
    assert "format" in varying
    assert "resolution" not in varying  # Same value for all
    assert "speed" not in varying


def test_get_worst_percentile_col() -> None:
    """Test worst percentile column selection."""
    # Higher is better metrics use p05 (lowest 5%)
    assert get_worst_percentile_col("ssimulacra2") == "p05"
    assert get_worst_percentile_col("psnr") == "p05"
    assert get_worst_percentile_col("ssim") == "p05"

    # Lower is better metrics use p95 (highest 5%)
    assert get_worst_percentile_col("butteraugli") == "p95"
    assert get_worst_percentile_col("bytes_per_pixel") == "p95"
    assert get_worst_percentile_col("bytes_per_ssimulacra2_per_pixel") == "p95"


def test_metric_directions() -> None:
    """Test metric direction definitions."""
    # Higher is better
    assert METRIC_DIRECTIONS["ssimulacra2"] is True
    assert METRIC_DIRECTIONS["psnr"] is True
    assert METRIC_DIRECTIONS["ssim"] is True

    # Lower is better
    assert METRIC_DIRECTIONS["butteraugli"] is False
    assert METRIC_DIRECTIONS["bytes_per_pixel"] is False
    assert METRIC_DIRECTIONS["bytes_per_butteraugli_per_pixel"] is False


def test_analyze_study_integration(tmp_path: Path, sample_quality_data: dict) -> None:
    """Test complete analysis workflow."""
    quality_file = tmp_path / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(sample_quality_data, f)

    output_dir = tmp_path / "analysis"

    # Run analysis
    analyze_study(quality_file, output_dir)

    # Check outputs exist
    assert output_dir.exists()

    # Check CSV exists
    csv_files = list(output_dir.glob("*_statistics.csv"))
    assert len(csv_files) == 1

    # Check some plots exist (SVG format)
    svg_files = list(output_dir.glob("*.svg"))
    assert len(svg_files) > 0

    # Verify CSV content
    stats_df = pd.read_csv(csv_files[0])
    assert len(stats_df) == 2  # Two quality levels
    assert "ssimulacra2_mean" in stats_df.columns


def test_efficiency_metric_calculation(sample_quality_data: dict) -> None:
    """Test encoder efficiency metric calculations."""
    df = create_analysis_dataframe(sample_quality_data)

    # bytes_per_ssimulacra2_per_pixel should be lower for better efficiency
    first = df.iloc[0]
    assert first["bytes_per_ssimulacra2_per_pixel"] > 0

    # Calculate manually for higher-is-better metric
    expected = first["bytes_per_pixel"] / first["ssimulacra2"]
    assert abs(first["bytes_per_ssimulacra2_per_pixel"] - expected) < 0.0001

    # For butteraugli (lower is better), efficiency is bytes * metric
    assert "bytes_per_butteraugli_per_pixel" in df.columns
    expected_butteraugli = first["bytes_per_pixel"] * first["butteraugli"]
    assert abs(first["bytes_per_butteraugli_per_pixel"] - expected_butteraugli) < 0.0001


def test_handles_null_metrics(sample_quality_data: dict) -> None:
    """Test handling of null/missing metrics."""
    # Add a measurement with null metrics
    sample_quality_data["measurements"].append(
        {
            "source_image": "data/datasets/test/image3.png",
            "original_image": "data/datasets/test/image3.png",
            "encoded_path": "data/encoded/test/avif/image3_q50.avif",
            "format": "avif",
            "quality": 50,
            "file_size": 90000,
            "width": 1920,
            "height": 1080,
            "source_file_size": 1000000,
            "ssimulacra2": None,
            "psnr": None,
            "ssim": None,
            "butteraugli": None,
            "chroma_subsampling": "420",
            "speed": 4,
            "resolution": None,
            "extra_args": None,
            "measurement_error": "Tool failed",
        }
    )

    df = create_analysis_dataframe(sample_quality_data)

    # Should not crash and should have NaN for efficiency metrics
    assert len(df) == 5
    assert pd.isna(df.iloc[-1]["bytes_per_ssimulacra2_per_pixel"])
