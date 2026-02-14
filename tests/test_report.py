"""Tests for report generation script."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_quality_data() -> dict:
    """Minimal quality data for report generation tests."""
    return {
        "study_id": "test-report",
        "study_name": "Test Report Study",
        "dataset": {"id": "test-ds", "path": "data/datasets/test", "image_count": 2},
        "encoding_timestamp": "2026-02-11T12:00:00+00:00",
        "timestamp": "2026-02-11T12:30:00+00:00",
        "measurements": [
            {
                "source_image": "img1.png",
                "original_image": "img1.png",
                "encoded_path": "avif/img1_q50.avif",
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
                "chroma_subsampling": None,
                "speed": None,
                "effort": None,
                "method": None,
                "resolution": None,
                "extra_args": None,
                "measurement_error": None,
            },
            {
                "source_image": "img1.png",
                "original_image": "img1.png",
                "encoded_path": "avif/img1_q80.avif",
                "format": "avif",
                "quality": 80,
                "file_size": 200000,
                "width": 1920,
                "height": 1080,
                "source_file_size": 1000000,
                "ssimulacra2": 85.0,
                "psnr": 42.0,
                "ssim": 0.98,
                "butteraugli": 1.5,
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


class TestReportGeneration:
    """Tests for generate_report.py functions."""

    def test_discover_studies_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test discovery with no metrics directory."""
        import scripts.generate_report as gen_report

        monkeypatch.setattr(gen_report, "METRICS_DIR", tmp_path / "nonexistent")
        result = gen_report.discover_studies()
        assert result == []

    def test_discover_studies_finds_quality_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_quality_data: dict
    ):
        """Test discovery finds studies with quality.json."""
        import scripts.generate_report as gen_report

        metrics_dir = tmp_path / "metrics"
        study_dir = metrics_dir / "test-study"
        study_dir.mkdir(parents=True)
        (study_dir / "quality.json").write_text(json.dumps(sample_quality_data))

        monkeypatch.setattr(gen_report, "METRICS_DIR", metrics_dir)
        result = gen_report.discover_studies()
        assert len(result) == 1
        assert result[0].name == "quality.json"

    def test_load_study_metadata(self, tmp_path: Path, sample_quality_data: dict):
        """Test loading metadata from quality.json."""
        from scripts.generate_report import _load_study_metadata

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))

        metadata = _load_study_metadata(quality_file)
        assert metadata["study_id"] == "test-report"
        assert metadata["study_name"] == "Test Report Study"
        assert metadata["image_count"] == 2
        assert metadata["measurement_count"] == 2
        assert "avif" in metadata["formats"]
        assert metadata["filename"] == "test-report.html"

    def test_figure_key_to_title(self):
        """Test converting figure keys to readable titles."""
        from scripts.generate_report import _figure_key_to_title

        title = _figure_key_to_title(
            "format-comparison_ssimulacra2_vs_quality", "format-comparison"
        )
        assert "SSIMULACRA2" in title
        assert "Quality" in title

        title = _figure_key_to_title("test_psnr_vs_bytes_per_pixel", "test")
        assert "PSNR" in title
        assert "Bytes Per Pixel" in title

    def test_generate_report_creates_files(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
    ):
        """Test that generate_report creates index.html and study pages."""
        from scripts.generate_report import generate_report

        # Set up quality.json
        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))

        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        assert (output_dir / "index.html").exists()
        assert (output_dir / "test-report.html").exists()
        assert (output_dir / "assets" / "plotly-basic.min.js").exists()

    def test_generated_html_contains_plotly_divs(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
    ):
        """Test that generated HTML contains interactive plot divs."""
        from scripts.generate_report import generate_report

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))
        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        study_html = (output_dir / "test-report.html").read_text()
        assert "plotly-graph-div" in study_html
        assert "plotly-basic.min.js" in study_html

    def test_generated_index_links_to_study(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
    ):
        """Test that index page links to study page."""
        from scripts.generate_report import generate_report

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))
        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        index_html = (output_dir / "index.html").read_text()
        assert "test-report.html" in index_html
        assert "Test Report Study" in index_html

    def test_multiple_studies(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
    ):
        """Test report with multiple studies."""
        from scripts.generate_report import generate_report

        # Create two study quality files
        q1 = tmp_path / "q1.json"
        q1.write_text(json.dumps(sample_quality_data))

        data2 = dict(sample_quality_data)
        data2["study_id"] = "second-study"
        data2["study_name"] = "Second Study"
        q2 = tmp_path / "q2.json"
        q2.write_text(json.dumps(data2))

        output_dir = tmp_path / "output"

        generate_report([q1, q2], output_dir)

        assert (output_dir / "index.html").exists()
        assert (output_dir / "test-report.html").exists()
        assert (output_dir / "second-study.html").exists()

        index_html = (output_dir / "index.html").read_text()
        assert "test-report.html" in index_html
        assert "second-study.html" in index_html
