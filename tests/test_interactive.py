"""Tests for interactive visualization module."""

import json
from pathlib import Path

import plotly.graph_objects as go
import pytest

from src.analysis import compute_statistics, create_analysis_dataframe
from src.interactive import (
    FORMAT_COLORS,
    METRIC_LABELS,
    PLOTLY_MARKERS,
    figure_to_html_fragment,
    generate_study_figures,
    plot_bytes_per_pixel,
    plot_efficiency,
    plot_quality_vs_param,
    plot_rate_distortion,
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


@pytest.fixture
def multi_format_quality_data() -> dict:
    """Create sample data with multiple formats for testing grouping."""
    measurements = []
    for fmt in ["jpeg", "webp", "avif"]:
        for q in [60, 80]:
            for img_idx in range(1, 3):
                measurements.append(
                    {
                        "source_image": f"data/datasets/test/image{img_idx}.png",
                        "original_image": f"data/datasets/test/image{img_idx}.png",
                        "encoded_path": f"data/encoded/test/{fmt}/image{img_idx}_q{q}.{fmt}",
                        "format": fmt,
                        "quality": q,
                        "file_size": 50000 + q * 1000 + img_idx * 5000,
                        "width": 1920,
                        "height": 1080,
                        "source_file_size": 1000000,
                        "ssimulacra2": 60.0 + q * 0.3 + img_idx,
                        "psnr": 30.0 + q * 0.1 + img_idx * 0.5,
                        "ssim": 0.85 + q * 0.001 + img_idx * 0.01,
                        "butteraugli": 4.0 - q * 0.03 + img_idx * 0.1,
                        "chroma_subsampling": None,
                        "speed": None,
                        "effort": None,
                        "method": None,
                        "resolution": None,
                        "extra_args": None,
                        "measurement_error": None,
                    }
                )

    return {
        "study_id": "multi-format-test",
        "study_name": "Multi-Format Test",
        "dataset": {
            "id": "test-dataset",
            "path": "data/datasets/test",
            "image_count": 2,
        },
        "encoding_timestamp": "2026-02-11T12:00:00+00:00",
        "timestamp": "2026-02-11T12:30:00+00:00",
        "measurements": measurements,
    }


@pytest.fixture
def single_quality_stats(sample_quality_data: dict):
    """Compute statistics from single-format sample data."""
    df = create_analysis_dataframe(sample_quality_data)
    return compute_statistics(df, ["quality"])


@pytest.fixture
def multi_format_stats(multi_format_quality_data: dict):
    """Compute statistics from multi-format sample data."""
    df = create_analysis_dataframe(multi_format_quality_data)
    return compute_statistics(df, ["format", "quality"])


class TestPlotQualityVsParam:
    """Tests for plot_quality_vs_param."""

    def test_returns_figure(self, single_quality_stats):
        fig = plot_quality_vs_param(single_quality_stats, "quality", "ssimulacra2")
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, single_quality_stats):
        fig = plot_quality_vs_param(single_quality_stats, "quality", "ssimulacra2")
        assert len(fig.data) >= 2  # mean + worst

    def test_grouped_traces(self, multi_format_stats):
        fig = plot_quality_vs_param(multi_format_stats, "quality", "ssimulacra2")
        # 3 formats × 2 traces (mean + worst) = 6 traces
        assert len(fig.data) == 6

    def test_missing_metric_returns_empty(self, single_quality_stats):
        fig = plot_quality_vs_param(single_quality_stats, "quality", "nonexistent_metric")
        assert len(fig.data) == 0

    def test_custom_title(self, single_quality_stats):
        fig = plot_quality_vs_param(
            single_quality_stats, "quality", "ssimulacra2", title="Custom Title"
        )
        assert fig.layout.title.text == "Custom Title"

    def test_traces_have_hover_text(self, single_quality_stats):
        fig = plot_quality_vs_param(single_quality_stats, "quality", "ssimulacra2")
        for trace in fig.data:
            assert trace.hovertext is not None
            assert len(trace.hovertext) > 0


class TestPlotRateDistortion:
    """Tests for plot_rate_distortion."""

    def test_returns_figure(self, single_quality_stats):
        fig = plot_rate_distortion(single_quality_stats, "ssimulacra2", None)
        assert isinstance(fig, go.Figure)

    def test_has_traces_without_grouping(self, single_quality_stats):
        fig = plot_rate_distortion(single_quality_stats, "ssimulacra2", None)
        assert len(fig.data) >= 2  # mean + worst

    def test_grouped_by_format(self, multi_format_stats):
        fig = plot_rate_distortion(multi_format_stats, "ssimulacra2", "format")
        # 3 formats × 2 traces = 6
        assert len(fig.data) == 6

    def test_missing_metric_returns_empty(self, single_quality_stats):
        fig = plot_rate_distortion(single_quality_stats, "nonexistent_metric", None)
        assert len(fig.data) == 0

    def test_x_axis_is_bytes_per_pixel(self, single_quality_stats):
        fig = plot_rate_distortion(single_quality_stats, "ssimulacra2", None)
        assert "Bytes per Pixel" in fig.layout.xaxis.title.text

    def test_hover_text_contains_quality(self, multi_format_stats):
        fig = plot_rate_distortion(multi_format_stats, "ssimulacra2", "format")
        # Check that hover text includes quality info
        for trace in fig.data:
            if trace.hovertext is not None and len(trace.hovertext) > 0:
                assert "Quality" in trace.hovertext[0]


class TestPlotBytesPerPixel:
    """Tests for plot_bytes_per_pixel."""

    def test_returns_figure(self, single_quality_stats):
        fig = plot_bytes_per_pixel(single_quality_stats, "quality")
        assert isinstance(fig, go.Figure)

    def test_has_mean_and_percentile_traces(self, single_quality_stats):
        fig = plot_bytes_per_pixel(single_quality_stats, "quality")
        # At minimum: mean, p05, p95
        assert len(fig.data) >= 3

    def test_missing_column_returns_empty(self):
        import pandas as pd

        empty_stats = pd.DataFrame({"quality": [50, 70]})
        fig = plot_bytes_per_pixel(empty_stats, "quality")
        assert len(fig.data) == 0


class TestPlotEfficiency:
    """Tests for plot_efficiency."""

    def test_returns_figure(self, single_quality_stats):
        fig = plot_efficiency(single_quality_stats, "quality", "bytes_per_ssimulacra2_per_pixel")
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, single_quality_stats):
        fig = plot_efficiency(single_quality_stats, "quality", "bytes_per_ssimulacra2_per_pixel")
        assert len(fig.data) >= 2  # mean + worst

    def test_missing_metric_returns_empty(self, single_quality_stats):
        fig = plot_efficiency(single_quality_stats, "quality", "nonexistent_metric")
        assert len(fig.data) == 0


class TestFigureToHtmlFragment:
    """Tests for figure_to_html_fragment."""

    def test_returns_html_string(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        html = figure_to_html_fragment(fig)
        assert isinstance(html, str)

    def test_contains_plotly_div(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        html = figure_to_html_fragment(fig)
        assert "plotly-graph-div" in html

    def test_does_not_include_plotlyjs(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        html = figure_to_html_fragment(fig)
        # Should not contain the plotly.js library inline
        assert "define&&define.amd" not in html

    def test_not_full_html(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        html = figure_to_html_fragment(fig)
        assert "<!DOCTYPE" not in html
        assert "<html" not in html


class TestGenerateStudyFigures:
    """Tests for generate_study_figures."""

    def test_generates_figures(self, tmp_path: Path, sample_quality_data: dict):
        quality_file = tmp_path / "quality.json"
        with open(quality_file, "w") as f:
            json.dump(sample_quality_data, f)

        figures = generate_study_figures(quality_file)
        assert isinstance(figures, dict)
        assert len(figures) > 0

    def test_figure_keys_contain_study_id(self, tmp_path: Path, sample_quality_data: dict):
        quality_file = tmp_path / "quality.json"
        with open(quality_file, "w") as f:
            json.dump(sample_quality_data, f)

        figures = generate_study_figures(quality_file)
        for key in figures:
            assert key.startswith("test-study_")

    def test_all_figures_are_plotly(self, tmp_path: Path, sample_quality_data: dict):
        quality_file = tmp_path / "quality.json"
        with open(quality_file, "w") as f:
            json.dump(sample_quality_data, f)

        figures = generate_study_figures(quality_file)
        for fig in figures.values():
            assert isinstance(fig, go.Figure)

    def test_includes_rate_distortion(self, tmp_path: Path, sample_quality_data: dict):
        quality_file = tmp_path / "quality.json"
        with open(quality_file, "w") as f:
            json.dump(sample_quality_data, f)

        figures = generate_study_figures(quality_file)
        bpp_keys = [k for k in figures if "bytes_per_pixel" in k]
        assert len(bpp_keys) > 0

    def test_includes_quality_metrics(self, tmp_path: Path, sample_quality_data: dict):
        quality_file = tmp_path / "quality.json"
        with open(quality_file, "w") as f:
            json.dump(sample_quality_data, f)

        figures = generate_study_figures(quality_file)
        metric_keys = [k for k in figures if "ssimulacra2" in k or "psnr" in k]
        assert len(metric_keys) > 0

    def test_multi_format_figures(self, tmp_path: Path, multi_format_quality_data: dict):
        quality_file = tmp_path / "quality.json"
        with open(quality_file, "w") as f:
            json.dump(multi_format_quality_data, f)

        figures = generate_study_figures(quality_file)
        assert len(figures) > 0

        # Rate-distortion should have grouped traces
        # With 3 formats and 2 qualities: x_param=format (3 unique),
        # secondary=quality (2 unique). Groups by quality: 2 groups × 2 traces = 4
        rd_keys = [k for k in figures if "ssimulacra2_vs_bytes_per_pixel" in k]
        if rd_keys:
            fig = figures[rd_keys[0]]
            assert len(fig.data) == 4

    def test_accepts_string_path(self, tmp_path: Path, sample_quality_data: dict):
        quality_file = tmp_path / "quality.json"
        with open(quality_file, "w") as f:
            json.dump(sample_quality_data, f)

        # Pass as string, not Path
        figures = generate_study_figures(str(quality_file))
        assert len(figures) > 0


class TestFormatColors:
    """Tests for format color assignments."""

    def test_all_formats_have_colors(self):
        for fmt in ["jpeg", "webp", "avif", "jxl"]:
            assert fmt in FORMAT_COLORS

    def test_colors_are_hex(self):
        for color in FORMAT_COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7


class TestConstants:
    """Tests for module constants."""

    def test_metric_labels_cover_main_metrics(self):
        for metric in ["ssimulacra2", "psnr", "ssim", "butteraugli"]:
            assert metric in METRIC_LABELS

    def test_plotly_markers_is_nonempty(self):
        assert len(PLOTLY_MARKERS) >= 8
