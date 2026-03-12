"""Tests for analysis module."""

import json
from pathlib import Path

import pandas as pd

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
    assert "bits_per_pixel" in df.columns
    assert "compression_ratio" in df.columns
    assert "bits_per_ssimulacra2_per_pixel" in df.columns

    # Check bits_per_pixel calculation
    # Image is 1920 * 1080 = 2,073,600 pixels
    expected_bpp_first = 8 * 100000 / (1920 * 1080)
    assert abs(df.iloc[0]["bits_per_pixel"] - expected_bpp_first) < 0.0001

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


def test_determine_sweep_parameter_crop() -> None:
    """Test sweep parameter detection with crop."""
    df = pd.DataFrame(
        {
            "quality": [50, 50, 50, 50],
            "crop": [2048, 1200, 800, 400],
            "format": ["avif", "avif", "avif", "avif"],
        }
    )

    sweep_param = determine_sweep_parameter(df)
    assert sweep_param == "crop"


def test_determine_varying_parameters() -> None:
    """Test detection of varying parameters."""
    df = pd.DataFrame(
        {
            "quality": [50, 60, 70, 80],
            "resolution": [1920, 1920, 1920, 1920],
            "format": ["avif", "avif", "jpeg", "jpeg"],
            "speed": [4, 4, 4, 4],
            "crop": [None, None, None, None],
        }
    )

    varying = determine_varying_parameters(df)
    assert "quality" in varying
    assert "format" in varying
    assert "resolution" not in varying  # Same value for all
    assert "speed" not in varying
    assert "crop" not in varying  # All None


def test_determine_varying_parameters_crop() -> None:
    """Test crop is detected as varying when it has multiple values."""
    df = pd.DataFrame(
        {
            "quality": [50, 50, 50],
            "crop": [800, 400, 200],
            "format": ["avif", "avif", "avif"],
        }
    )

    varying = determine_varying_parameters(df)
    assert "crop" in varying
    assert "quality" not in varying


def test_get_worst_percentile_col() -> None:
    """Test worst percentile column selection."""
    # Higher is better metrics use p05 (lowest 5%)
    assert get_worst_percentile_col("ssimulacra2") == "p05"
    assert get_worst_percentile_col("psnr") == "p05"
    assert get_worst_percentile_col("ssim") == "p05"

    # Lower is better metrics use p95 (highest 5%)
    assert get_worst_percentile_col("butteraugli") == "p95"
    assert get_worst_percentile_col("bits_per_pixel") == "p95"
    assert get_worst_percentile_col("bits_per_ssimulacra2_per_pixel") == "p95"


def test_metric_directions() -> None:
    """Test metric direction definitions."""
    # Higher is better
    assert METRIC_DIRECTIONS["ssimulacra2"] is True
    assert METRIC_DIRECTIONS["psnr"] is True
    assert METRIC_DIRECTIONS["ssim"] is True

    # Lower is better
    assert METRIC_DIRECTIONS["butteraugli"] is False
    assert METRIC_DIRECTIONS["bits_per_pixel"] is False
    assert METRIC_DIRECTIONS["bits_per_butteraugli_per_pixel"] is False


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

    # bits_per_ssimulacra2_per_pixel should be lower for better efficiency
    first = df.iloc[0]
    assert first["bits_per_ssimulacra2_per_pixel"] > 0

    # Calculate manually for higher-is-better metric
    expected = first["bits_per_pixel"] / first["ssimulacra2"]
    assert abs(first["bits_per_ssimulacra2_per_pixel"] - expected) < 0.0001

    # For butteraugli (lower is better), efficiency is bits * metric
    assert "bits_per_butteraugli_per_pixel" in df.columns
    expected_butteraugli = first["bits_per_pixel"] * first["butteraugli"]
    assert abs(first["bits_per_butteraugli_per_pixel"] - expected_butteraugli) < 0.0001


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
    assert pd.isna(df.iloc[-1]["bits_per_ssimulacra2_per_pixel"])


# ---------------------------------------------------------------------------
# analyze_study with explicit x_axis / group_by
# ---------------------------------------------------------------------------


def _make_multi_param_quality_data() -> dict:
    """Build a quality dataset with two varying params: quality and speed."""
    measurements = []
    for quality in [50, 70]:
        for speed in [0, 4, 8]:
            measurements.append(
                {
                    "source_image": "data/datasets/test/image1.png",
                    "original_image": "data/datasets/test/image1.png",
                    "encoded_path": f"data/encoded/test/avif/img_q{quality}_s{speed}.avif",
                    "format": "avif",
                    "quality": quality,
                    "speed": speed,
                    "file_size": 100_000 + quality * 200 + speed * 50,
                    "width": 640,
                    "height": 480,
                    "source_file_size": 1_000_000,
                    "ssimulacra2": 70.0 + quality * 0.2 - speed * 0.5,
                    "psnr": 38.0,
                    "ssim": 0.95,
                    "butteraugli": 2.5,
                    "chroma_subsampling": None,
                    "effort": None,
                    "method": None,
                    "resolution": None,
                    "extra_args": None,
                    "measurement_error": None,
                }
            )
    return {
        "study_id": "speed-study",
        "study_name": "Speed Study",
        "dataset": {"id": "test", "path": "data/datasets/test", "image_count": 1},
        "encoding_timestamp": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "measurements": measurements,
    }


def test_analyze_study_explicit_x_axis(tmp_path: Path) -> None:
    """analyze_study uses the explicit x_axis parameter when supplied."""
    data = _make_multi_param_quality_data()
    quality_file = tmp_path / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(data, f)

    output_dir = tmp_path / "analysis"
    analyze_study(quality_file, output_dir, x_axis="speed")

    # A plot file named *_vs_speed.svg should be generated
    svg_files = list(output_dir.glob("*_vs_speed.svg"))
    assert len(svg_files) > 0, "Expected at least one plot with 'speed' x-axis"


def test_analyze_study_explicit_group_by(tmp_path: Path) -> None:
    """analyze_study uses the explicit group_by when supplied."""
    data = _make_multi_param_quality_data()
    quality_file = tmp_path / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(data, f)

    output_dir = tmp_path / "analysis"
    # x_axis=speed, group_by=quality: should not raise
    analyze_study(quality_file, output_dir, x_axis="speed", group_by="quality")

    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) == 1


def test_analyze_study_reads_metadata_x_axis(tmp_path: Path) -> None:
    """analyze_study picks up analysis_x_axis from quality.json metadata."""
    data = _make_multi_param_quality_data()
    data["analysis_x_axis"] = "speed"
    data["analysis_group_by"] = "quality"
    quality_file = tmp_path / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(data, f)

    output_dir = tmp_path / "analysis"
    analyze_study(quality_file, output_dir)

    svg_files = list(output_dir.glob("*_vs_speed.svg"))
    assert len(svg_files) > 0, "Expected plot named *_vs_speed.svg from metadata"


def test_analyze_study_cli_overrides_metadata(tmp_path: Path) -> None:
    """Explicit CLI x_axis overrides analysis_x_axis stored in quality.json."""
    data = _make_multi_param_quality_data()
    data["analysis_x_axis"] = "quality"  # metadata says quality
    quality_file = tmp_path / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(data, f)

    output_dir = tmp_path / "analysis"
    # Explicit x_axis=speed should win over metadata
    analyze_study(quality_file, output_dir, x_axis="speed")

    svg_files = list(output_dir.glob("*_vs_speed.svg"))
    assert len(svg_files) > 0, "Expected plot with 'speed' axis despite metadata saying 'quality'"


def test_analyze_study_falls_back_to_heuristic(tmp_path: Path) -> None:
    """analyze_study falls back to heuristic when neither CLI nor metadata is set."""
    data = _make_multi_param_quality_data()
    # No analysis_x_axis in data → heuristic picks param with most unique values
    quality_file = tmp_path / "quality.json"
    with open(quality_file, "w") as f:
        json.dump(data, f)

    output_dir = tmp_path / "analysis"
    analyze_study(quality_file, output_dir)

    # Speed has 3 unique values, quality has 2 → heuristic selects speed
    svg_files = list(output_dir.glob("*_vs_speed.svg"))
    assert len(svg_files) > 0
