"""Tests for the report image optimisation module."""

from pathlib import Path

import pytest
from PIL import Image

from src.report_images import (
    SVG_PRECISION,
    ImageVariant,
    OptimisedImage,
    StudyComparisonImages,
    _classify_image,
    discover_and_optimise,
    img_srcset_html,
    optimise_lossless,
    optimise_lossy,
    optimise_svg,
    picture_html,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_report_root(tmp_path: Path) -> Path:
    """Create a temporary report output root."""
    d = tmp_path / "report"
    d.mkdir()
    return d


@pytest.fixture()
def sample_lossless_grid(tmp_path: Path) -> Path:
    """Create a synthetic lossless WebP image imitating a 3× comparison grid.

    4 cells of 384×384 arranged in a 2×2 grid with 4px gaps and 66px label
    headers → roughly (384*2 + 4*3) × (384+66)*2 + 4*3 = 780 × 912.
    We use a simpler 768×900 for testing purposes.
    """
    img = Image.new("RGB", (768, 900), color=(128, 64, 200))
    p = tmp_path / "comparison_format_avif.webp"
    img.save(p, format="WEBP", lossless=True)
    return p


@pytest.fixture()
def sample_lossy_source(tmp_path: Path) -> Path:
    """Create a sample source image for lossy optimisation."""
    img = Image.new("RGB", (2040, 1128), color=(100, 150, 50))
    p = tmp_path / "original_annotated.webp"
    img.save(p, format="WEBP", lossless=True)
    return p


@pytest.fixture()
def comparison_tree(tmp_path: Path) -> Path:
    """Build a minimal comparison output directory tree.

    Structure::

        <tmp>/analysis/<study>/comparison/
            average/
                comparison.webp
                distortion_map_average.webp
                distortion_map_comparison.webp
                original_annotated.webp
            variance/
                comparison.webp
                distortion_map_variance.webp
                distortion_map_comparison.webp
                original_annotated.webp
    """
    analysis = tmp_path / "analysis"
    study_id = "test-study"
    for strategy in ("average", "variance"):
        d = analysis / study_id / "comparison" / strategy
        d.mkdir(parents=True)
        for name in (
            "comparison.webp",
            f"distortion_map_{strategy}.webp",
            "distortion_map_comparison.webp",
            "original_annotated.webp",
        ):
            img = Image.new("RGB", (384, 450), color=(80, 80, 80))
            img.save(d / name, format="WEBP", lossless=True)
    return analysis


@pytest.fixture()
def comparison_tree_with_resolution(tmp_path: Path) -> Path:
    """Build a comparison tree with per-resolution subdirectories."""
    analysis = tmp_path / "analysis"
    study_id = "res-study"
    for strategy in ("average",):
        for res in ("r120", "r240"):
            d = analysis / study_id / "comparison" / strategy / res
            d.mkdir(parents=True)
            for name in (
                "comparison.webp",
                f"distortion_map_{strategy}.webp",
                "distortion_map_comparison.webp",
                "original_annotated.webp",
            ):
                img = Image.new("RGB", (384, 450), color=(80, 80, 80))
                img.save(d / name, format="WEBP", lossless=True)
    return analysis


@pytest.fixture()
def sample_svg(tmp_path: Path) -> Path:
    """Create a minimal SVG imitating Matplotlib output.

    Includes:
    - 6-decimal-place coordinates (to verify precision reduction)
    - A ``<style>`` block (to verify ``removeStyleElement`` is applied)
    """
    svg_content = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="400.123456" height="300.654321">\n'
        "  <style>rect { fill: red; } line { stroke: blue; }</style>\n"
        '  <rect x="10.123456" y="10.654321" width="380.246813" height="280.975309" fill="red"/>\n'
        '  <line x1="0.111111" y1="0.222222" x2="399.888889" y2="299.777778" stroke="blue"/>\n'
        "</svg>\n"
    )
    p = tmp_path / "figure.svg"
    p.write_text(svg_content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# optimise_svg
# ---------------------------------------------------------------------------


class TestOptimiseSvg:
    def test_creates_output_file(self, sample_svg: Path, tmp_path: Path) -> None:
        dest = tmp_path / "out" / "figure.svg"
        result = optimise_svg(sample_svg, dest)
        assert result == dest
        assert dest.exists()
        assert dest.stat().st_size > 0

    def test_output_is_valid_svg(self, sample_svg: Path, tmp_path: Path) -> None:
        dest = tmp_path / "figure_opt.svg"
        optimise_svg(sample_svg, dest)
        content = dest.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "</svg>" in content

    def test_output_is_smaller_or_equal(self, sample_svg: Path, tmp_path: Path) -> None:
        dest = tmp_path / "figure_opt.svg"
        optimise_svg(sample_svg, dest)
        assert dest.stat().st_size <= sample_svg.stat().st_size

    def test_precision_applied(self, sample_svg: Path, tmp_path: Path) -> None:
        """Values like 10.123456 should be rounded to 1 decimal place."""
        import re

        dest = tmp_path / "figure_opt.svg"
        optimise_svg(sample_svg, dest, precision=1)
        content = dest.read_text(encoding="utf-8")
        # No multi-digit decimal runs should remain
        assert not re.search(r"\d+\.\d{3,}", content), (
            "Expected precision to be reduced to ≤2 decimal places"
        )

    def test_style_element_removed(self, sample_svg: Path, tmp_path: Path) -> None:
        """The ``<style>`` block should be stripped by ``removeStyleElement``."""
        dest = tmp_path / "figure_opt.svg"
        optimise_svg(sample_svg, dest)
        content = dest.read_text(encoding="utf-8")
        assert "<style" not in content

    def test_creates_parent_directories(self, sample_svg: Path, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c" / "figure.svg"
        optimise_svg(sample_svg, dest)
        assert dest.exists()

    def test_default_precision_is_svg_precision_constant(self) -> None:
        """SVG_PRECISION constant value is 1 (sufficient for screen rendering)."""
        assert SVG_PRECISION == 1


# ---------------------------------------------------------------------------


class TestClassifyImage:
    def test_fragment_grid(self) -> None:
        assert _classify_image("comparison.webp") == "fragment_grid"
        assert _classify_image("comparison_format_avif.webp") == "fragment_grid"

    def test_distmap_grid(self) -> None:
        assert _classify_image("distortion_map_comparison.webp") == "distmap_grid"
        assert _classify_image("distortion_map_comparison_format_jpeg.webp") == "distmap_grid"

    def test_distortion_map(self) -> None:
        assert _classify_image("distortion_map_average.webp") == "distortion_map"
        assert _classify_image("distortion_map_variance.webp") == "distortion_map"

    def test_original_annotated(self) -> None:
        assert _classify_image("original_annotated.webp") == "original_annotated"

    def test_unknown(self) -> None:
        assert _classify_image("some_random_file.webp") == "unknown"


# ---------------------------------------------------------------------------
# optimise_lossless
# ---------------------------------------------------------------------------


class TestOptimiseLossless:
    def test_creates_single_variant(
        self, sample_lossless_grid: Path, tmp_report_root: Path
    ) -> None:
        out_dir = tmp_report_root / "fragments"
        opt = optimise_lossless(sample_lossless_grid, out_dir, tmp_report_root, alt="test grid")
        assert opt.kind == "lossless"
        webp = opt.variants["image/webp"]
        assert len(webp) == 1
        assert webp[0].width == 768  # original 3× width preserved
        assert webp[0].height == 900
        assert webp[0].lossless is True

    def test_default_is_full_resolution(
        self, sample_lossless_grid: Path, tmp_report_root: Path
    ) -> None:
        out_dir = tmp_report_root / "fragments"
        opt = optimise_lossless(sample_lossless_grid, out_dir, tmp_report_root, alt="test")
        assert opt.default is not None
        assert opt.default.width == 768

    def test_relative_paths(self, sample_lossless_grid: Path, tmp_report_root: Path) -> None:
        out_dir = tmp_report_root / "fragments"
        opt = optimise_lossless(sample_lossless_grid, out_dir, tmp_report_root, alt="test")
        for v in opt.variants["image/webp"]:
            assert not v.rel_path.startswith("/")
            assert "fragments" in v.rel_path


# ---------------------------------------------------------------------------
# optimise_lossy
# ---------------------------------------------------------------------------


class TestOptimiseLossy:
    def test_creates_avif_and_webp(self, sample_lossy_source: Path, tmp_report_root: Path) -> None:
        out_dir = tmp_report_root / "overview"
        opt = optimise_lossy(sample_lossy_source, out_dir, tmp_report_root, alt="test lossy")
        assert opt.kind == "lossy"
        assert "image/avif" in opt.variants
        assert "image/webp" in opt.variants
        # Should have multiple widths
        assert len(opt.variants["image/avif"]) >= 2
        assert len(opt.variants["image/webp"]) >= 2

    def test_includes_original_width(
        self, sample_lossy_source: Path, tmp_report_root: Path
    ) -> None:
        out_dir = tmp_report_root / "overview"
        opt = optimise_lossy(sample_lossy_source, out_dir, tmp_report_root, alt="test")
        avif_widths = [v.width for v in opt.variants["image/avif"]]
        assert 2040 in avif_widths

    def test_no_upscaling(self, tmp_path: Path, tmp_report_root: Path) -> None:
        """If source is smaller than a target width, skip that width."""
        small = tmp_path / "small.webp"
        Image.new("RGB", (500, 300)).save(small, format="WEBP")
        out_dir = tmp_report_root / "overview"
        opt = optimise_lossy(small, out_dir, tmp_report_root, alt="small", target_widths=[600, 900])
        # Only original width (500) should be present
        widths = [v.width for v in opt.variants["image/avif"]]
        assert all(w <= 500 for w in widths)

    def test_default_is_middle_webp(self, sample_lossy_source: Path, tmp_report_root: Path) -> None:
        out_dir = tmp_report_root / "overview"
        opt = optimise_lossy(sample_lossy_source, out_dir, tmp_report_root, alt="test")
        assert opt.default is not None
        assert opt.default.media_type == "image/webp"


# ---------------------------------------------------------------------------
# discover_and_optimise
# ---------------------------------------------------------------------------


class TestDiscoverAndOptimise:
    def test_discovers_both_strategies(self, comparison_tree: Path, tmp_report_root: Path) -> None:
        result = discover_and_optimise(
            comparison_tree, "test-study", tmp_report_root, tmp_report_root
        )
        assert isinstance(result, StudyComparisonImages)
        assert result.study_id == "test-study"
        assert len(result.sets) == 2  # average + variance
        strategies = {s.strategy for s in result.sets}
        assert strategies == {"average", "variance"}

    def test_each_set_has_all_figure_types(
        self, comparison_tree: Path, tmp_report_root: Path
    ) -> None:
        result = discover_and_optimise(
            comparison_tree, "test-study", tmp_report_root, tmp_report_root
        )
        for img_set in result.sets:
            assert img_set.original_annotated is not None
            assert img_set.distortion_map is not None
            assert len(img_set.fragment_grids) >= 1
            assert len(img_set.distmap_grids) >= 1

    def test_no_comparison_dir_returns_empty(self, tmp_path: Path, tmp_report_root: Path) -> None:
        analysis = tmp_path / "analysis"
        analysis.mkdir()
        result = discover_and_optimise(analysis, "nonexistent", tmp_report_root, tmp_report_root)
        assert result.sets == []

    def test_resolution_subdirectories(
        self, comparison_tree_with_resolution: Path, tmp_report_root: Path
    ) -> None:
        result = discover_and_optimise(
            comparison_tree_with_resolution,
            "res-study",
            tmp_report_root,
            tmp_report_root,
        )
        assert len(result.sets) == 2  # r120 + r240 under average
        resolutions = {s.resolution for s in result.sets}
        assert resolutions == {"r120", "r240"}
        for s in result.sets:
            assert s.strategy == "average"


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


class TestHtmlHelpers:
    def test_img_srcset_html(self) -> None:
        opt = OptimisedImage(source_path=Path("dummy.webp"), alt="test", kind="lossless")
        opt.variants["image/webp"] = [
            ImageVariant(Path("a.webp"), "img/a_1x.webp", 256, 300, "image/webp", True),
            ImageVariant(Path("b.webp"), "img/a_2x.webp", 512, 600, "image/webp", True),
            ImageVariant(Path("c.webp"), "img/a_3x.webp", 768, 900, "image/webp", True),
        ]
        opt.default = opt.variants["image/webp"][1]
        html = img_srcset_html(opt, css_class="my-class")
        assert "srcset=" in html
        assert "256w" in html
        assert "512w" in html
        assert "768w" in html
        assert 'class="my-class"' in html
        assert 'loading="lazy"' in html
        assert 'src="img/a_2x.webp"' in html

    def test_picture_html(self) -> None:
        opt = OptimisedImage(source_path=Path("dummy.webp"), alt="test lossy", kind="lossy")
        opt.variants["image/avif"] = [
            ImageVariant(Path("a.avif"), "img/a_600.avif", 600, 330, "image/avif"),
            ImageVariant(Path("b.avif"), "img/a_900.avif", 900, 497, "image/avif"),
        ]
        opt.variants["image/webp"] = [
            ImageVariant(Path("a.webp"), "img/a_600.webp", 600, 330, "image/webp"),
            ImageVariant(Path("b.webp"), "img/a_900.webp", 900, 497, "image/webp"),
        ]
        opt.default = opt.variants["image/webp"][0]
        html = picture_html(opt)
        assert "<picture>" in html
        assert "</picture>" in html
        assert 'type="image/avif"' in html
        assert 'type="image/webp"' in html
        assert "600w" in html
        assert "900w" in html
        assert 'loading="lazy"' in html
        assert 'alt="test lossy"' in html

    def test_picture_html_with_custom_sizes(self) -> None:
        opt = OptimisedImage(source_path=Path("dummy.webp"), alt="sized", kind="lossy")
        opt.variants["image/webp"] = [
            ImageVariant(Path("a.webp"), "a.webp", 400, 300, "image/webp"),
        ]
        opt.default = opt.variants["image/webp"][0]
        html = picture_html(opt, sizes="100vw")
        assert 'sizes="100vw"' in html
