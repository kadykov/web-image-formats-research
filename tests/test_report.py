"""Tests for report generation script."""

import json
from pathlib import Path
from xml.etree.ElementTree import parse as parse_xml

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


@pytest.fixture
def _mock_report_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Mock report asset downloads so tests don't require network access."""

    import scripts.generate_report as gen_report

    def _make_dummy(name: str) -> Path:
        p = tmp_path / name
        p.write_text(f"// dummy {name}\n")
        return p

    def _ensure_plotly_bundle() -> Path:
        return _make_dummy("plotly-basic.min.js")

    def _ensure_photoswipe_bundle() -> tuple[Path, Path, Path]:
        return (
            _make_dummy("photoswipe.css"),
            _make_dummy("photoswipe.esm.min.js"),
            _make_dummy("photoswipe-lightbox.esm.min.js"),
        )

    monkeypatch.setattr(gen_report, "ensure_plotly_bundle", _ensure_plotly_bundle)
    monkeypatch.setattr(gen_report, "ensure_photoswipe_bundle", _ensure_photoswipe_bundle)
    return tmp_path


def _sitemap_locs(sitemap_path: Path) -> list[str]:
    """Return all <loc> values from a sitemap XML file.

    Handles cases where the sitemap root element uses an XML namespace.
    """

    tree = parse_xml(sitemap_path)
    root = tree.getroot()

    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
        namespace = {"ns": ns.strip("{}")}
        return [elem.text for elem in root.findall(".//ns:loc", namespace)]

    return [elem.text for elem in root.findall(".//loc")]


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

        title = _figure_key_to_title("test_psnr_vs_bits_per_pixel", "test")
        assert "PSNR" in title
        assert "BPP" in title

    def test_generate_report_creates_files(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
        _mock_report_assets: Path,
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
        _mock_report_assets: Path,
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
        _mock_report_assets: Path,
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
        _mock_report_assets: Path,
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

    def test_sitemap_created(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
        _mock_report_assets: Path,
    ):
        """Test that sitemap.xml is created."""
        from scripts.generate_report import generate_report

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))
        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        assert (output_dir / "sitemap.xml").exists()

    def test_sitemap_valid_xml(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
        _mock_report_assets: Path,
    ):
        """Test that sitemap.xml is valid XML."""
        from scripts.generate_report import generate_report

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))
        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        sitemap_path = output_dir / "sitemap.xml"
        tree = parse_xml(sitemap_path)
        root = tree.getroot()
        assert root.tag.endswith("urlset")

    def test_sitemap_contains_index(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
        _mock_report_assets: Path,
    ):
        """Test that sitemap includes index.html with proper path."""
        from scripts.generate_report import generate_report

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))
        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        # Verify sitemap has valid entries including index
        locs = _sitemap_locs(output_dir / "sitemap.xml")
        assert len(locs) >= 2  # At least index and study page
        # One entry should be a root path (index.html)
        assert any(loc.endswith("/") for loc in locs)

    def test_sitemap_contains_study_pages(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
        _mock_report_assets: Path,
    ):
        """Test that sitemap includes study pages."""
        from scripts.generate_report import generate_report

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))
        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        # Should contain reference to study HTML file
        assert "test-report.html" in sitemap_text

    def test_sitemap_excludes_404(
        self,
        tmp_path: Path,
    ):
        """Test that sitemap excludes 404.html."""
        from scripts.generate_report import _write_report_sitemap

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create test HTML files
        (output_dir / "index.html").write_text("<html><body>Index</body></html>")
        (output_dir / "study.html").write_text("<html><body>Study</body></html>")
        (output_dir / "404.html").write_text("<html><body>Not Found</body></html>")

        _write_report_sitemap(output_dir, study_filenames=["study.html"])

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        assert "404.html" not in sitemap_text

    def test_sitemap_excludes_noindex(
        self,
        tmp_path: Path,
    ):
        """Test that sitemap excludes pages with noindex directive."""
        from scripts.generate_report import _write_report_sitemap

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create test HTML files
        (output_dir / "index.html").write_text("<html><body>Index</body></html>")
        (output_dir / "excluded.html").write_text(
            '<html><head><meta name="robots" content="noindex,follow"></head><body>Excluded</body></html>'
        )

        _write_report_sitemap(output_dir, study_filenames=["excluded.html"])

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        assert "excluded.html" not in sitemap_text

    def test_sitemap_includes_pages_without_noindex(
        self,
        tmp_path: Path,
    ):
        """Test that sitemap includes pages without noindex directive (default behavior)."""
        from scripts.generate_report import _write_report_sitemap

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create test HTML files
        (output_dir / "index.html").write_text("<html><body>Index</body></html>")
        (output_dir / "study1.html").write_text("<html><body>Study 1</body></html>")
        (output_dir / "study2.html").write_text(
            '<html><head><meta name="robots" content="index,follow"></head><body>Study 2</body></html>'
        )

        _write_report_sitemap(output_dir, study_filenames=["study1.html", "study2.html"])

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        # Both pages without noindex should be included
        assert "study1.html" in sitemap_text
        assert "study2.html" in sitemap_text
        # Verify using locs helper for more robust check
        locs = _sitemap_locs(output_dir / "sitemap.xml")
        assert len(locs) >= 2  # index + study1 + study2

    def test_sitemap_multiple_studies(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
        _mock_report_assets: Path,
    ):
        """Test that sitemap includes all study pages from multiple studies."""
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

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        assert "test-report.html" in sitemap_text
        assert "second-study.html" in sitemap_text

    def test_sitemap_xml_structure(
        self,
        tmp_path: Path,
        sample_quality_data: dict,
        _mock_report_assets: Path,
    ):
        """Test that sitemap has proper XML structure with url elements."""
        from scripts.generate_report import generate_report

        quality_file = tmp_path / "quality.json"
        quality_file.write_text(json.dumps(sample_quality_data))
        output_dir = tmp_path / "output"

        generate_report([quality_file], output_dir)

        locs = _sitemap_locs(output_dir / "sitemap.xml")

        # Should have at least index and study pages
        assert len(locs) >= 2

    def test_sitemap_with_study_filenames_only_includes_specified(
        self,
        tmp_path: Path,
    ):
        """Test that sitemap with study_filenames parameter only includes specified pages."""
        from scripts.generate_report import _write_report_sitemap

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create multiple HTML files
        (output_dir / "index.html").write_text("<html><body>Index</body></html>")
        (output_dir / "study1.html").write_text("<html><body>Study 1</body></html>")
        (output_dir / "study2.html").write_text("<html><body>Study 2</body></html>")
        (output_dir / "study3.html").write_text("<html><body>Study 3</body></html>")

        # Only include study1 and study2 in the sitemap
        _write_report_sitemap(output_dir, study_filenames=["study1.html", "study2.html"])

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        assert "study1.html" in sitemap_text
        assert "study2.html" in sitemap_text
        # study3 should NOT be in the sitemap even though it exists in the directory
        assert "study3.html" not in sitemap_text

    def test_sitemap_without_study_filenames_scans_directory(
        self,
        tmp_path: Path,
    ):
        """Test that sitemap without study_filenames scans directory (legacy behavior)."""
        from scripts.generate_report import _write_report_sitemap

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create multiple HTML files
        (output_dir / "index.html").write_text("<html><body>Index</body></html>")
        (output_dir / "study1.html").write_text("<html><body>Study 1</body></html>")
        (output_dir / "study2.html").write_text("<html><body>Study 2</body></html>")

        # Call without study_filenames - should scan directory
        _write_report_sitemap(output_dir, study_filenames=None)

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        # All files should be in the sitemap
        assert "study1.html" in sitemap_text
        assert "study2.html" in sitemap_text

    def test_sitemap_with_study_filenames_includes_index(
        self,
        tmp_path: Path,
    ):
        """Test that sitemap includes index.html automatically with study_filenames."""
        from scripts.generate_report import _write_report_sitemap

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        (output_dir / "index.html").write_text("<html><body>Index</body></html>")
        (output_dir / "study1.html").write_text("<html><body>Study 1</body></html>")

        _write_report_sitemap(output_dir, study_filenames=["study1.html"])

        locs = _sitemap_locs(output_dir / "sitemap.xml")

        # Should have at least 2 entries: index and study1
        assert len(locs) >= 2
        # At least one should be the root path (for index.html)
        assert any("/" in loc and "study1" not in loc for loc in locs)

    def test_sitemap_generation_handles_missing_files(
        self,
        tmp_path: Path,
    ):
        """Test that sitemap generation handles gracefully when specified files don't exist."""
        from scripts.generate_report import _write_report_sitemap

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        (output_dir / "index.html").write_text("<html><body>Index</body></html>")
        (output_dir / "study1.html").write_text("<html><body>Study 1</body></html>")

        # Request a file that doesn't exist
        _write_report_sitemap(output_dir, study_filenames=["study1.html", "nonexistent.html"])

        sitemap_text = (output_dir / "sitemap.xml").read_text()
        # Should have study1 but not the nonexistent file
        assert "study1.html" in sitemap_text
        assert "nonexistent" not in sitemap_text
