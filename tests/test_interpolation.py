"""Tests for the interpolation module."""

from __future__ import annotations

import pytest

from src.interpolation import (
    _collect_quality_metric_pairs,
    _interpolate_target,
    _lerp,
    compute_cross_format_cv,
    interpolate_metric_at_quality,
    interpolate_quality_for_metric,
    select_best_image,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_M = {
    "width": 100,
    "height": 100,
    "measurement_error": None,
    "ssimulacra2": 70.0,
    "speed": None,
    "effort": None,
    "method": None,
    "chroma_subsampling": None,
    "resolution": None,
}


def _make_m(fmt: str, quality: int, ssim: float, file_size: int, image: str = "img.png") -> dict:
    return {
        **_BASE_M,
        "format": fmt,
        "quality": quality,
        "ssimulacra2": ssim,
        "file_size": file_size,
        "source_image": image,
        "original_image": image,
    }


# ---------------------------------------------------------------------------
# _lerp
# ---------------------------------------------------------------------------


class TestLerp:
    def test_basic(self) -> None:
        assert _lerp(0.0, 0.0, 1.0, 1.0, 0.5) == pytest.approx(0.5)

    def test_equal_y_returns_midpoint_x(self) -> None:
        assert _lerp(10.0, 5.0, 20.0, 5.0, 5.0) == pytest.approx(15.0)

    def test_descending(self) -> None:
        # quality 100→0 as metric goes from 90→40; target metric=60 → halfway
        assert _lerp(100.0, 90.0, 0.0, 40.0, 60.0) == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# _collect_quality_metric_pairs
# ---------------------------------------------------------------------------


class TestCollectQualityMetricPairs:
    def test_filters_by_format(self) -> None:
        ms = [_make_m("jpeg", 50, 55.0, 5000), _make_m("avif", 50, 70.0, 4000)]
        pairs = _collect_quality_metric_pairs(ms, "jpeg", "ssimulacra2")
        assert len(pairs) == 1
        assert pairs[0][1] == pytest.approx(55.0)

    def test_skips_measurement_error(self) -> None:
        m = {**_make_m("jpeg", 50, 55.0, 5000), "measurement_error": "some error"}
        pairs = _collect_quality_metric_pairs([m], "jpeg", "ssimulacra2")
        assert pairs == []

    def test_bytes_per_pixel(self) -> None:
        # 100×100 image, file_size=1000 → bpp = 0.1
        ms = [_make_m("jpeg", 50, 55.0, 1000), _make_m("jpeg", 80, 75.0, 2000)]
        pairs = _collect_quality_metric_pairs(ms, "jpeg", "bytes_per_pixel")
        assert pairs[0][1] == pytest.approx(0.1)
        assert pairs[1][1] == pytest.approx(0.2)

    def test_averages_duplicate_quality_levels(self) -> None:
        ms = [
            _make_m("jpeg", 50, 60.0, 5000, "img1.png"),
            _make_m("jpeg", 50, 80.0, 5000, "img2.png"),
        ]
        pairs = _collect_quality_metric_pairs(ms, "jpeg", "ssimulacra2")
        assert len(pairs) == 1
        assert pairs[0][1] == pytest.approx(70.0)

    def test_sorted_by_quality(self) -> None:
        ms = [_make_m("jpeg", 80, 80.0, 8000), _make_m("jpeg", 30, 40.0, 3000)]
        pairs = _collect_quality_metric_pairs(ms, "jpeg", "ssimulacra2")
        assert pairs[0][0] < pairs[1][0]


# ---------------------------------------------------------------------------
# _interpolate_target
# ---------------------------------------------------------------------------


class TestInterpolateTarget:
    def test_midpoint(self) -> None:
        points = [(0.0, 0.0), (100.0, 100.0)]
        assert _interpolate_target(points, 50.0) == pytest.approx(50.0)

    def test_out_of_range_returns_none(self) -> None:
        points = [(0.0, 0.0), (100.0, 100.0)]
        assert _interpolate_target(points, 110.0) is None

    def test_descending_metric(self) -> None:
        # bytes-per-pixel decreases as quality setting increases (unusual but valid)
        points = [(30.0, 0.5), (80.0, 0.1)]
        # target 0.3 is between 0.5 and 0.1 → quality ~55
        result = _interpolate_target(points, 0.3)
        assert result is not None
        assert 30.0 < result < 80.0


# ---------------------------------------------------------------------------
# interpolate_quality_for_metric / interpolate_metric_at_quality
# ---------------------------------------------------------------------------


class TestInterpolateQualityForMetric:
    def test_basic_interpolation(self) -> None:
        ms = [_make_m("jpeg", 50, 50.0, 5000), _make_m("jpeg", 90, 90.0, 9000)]
        result = interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 70.0)
        assert result == pytest.approx(70.0)

    def test_returns_none_out_of_range(self) -> None:
        ms = [_make_m("jpeg", 50, 50.0, 5000), _make_m("jpeg", 90, 90.0, 9000)]
        assert interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 95.0) is None

    def test_returns_none_insufficient_data(self) -> None:
        ms = [_make_m("jpeg", 50, 50.0, 5000)]
        assert interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 50.0) is None


class TestInterpolateMetricAtQuality:
    def test_basic_interpolation(self) -> None:
        ms = [_make_m("jpeg", 40, 40.0, 4000), _make_m("jpeg", 80, 80.0, 8000)]
        result = interpolate_metric_at_quality(ms, "jpeg", 60.0, "ssimulacra2")
        assert result == pytest.approx(60.0)

    def test_returns_none_out_of_range(self) -> None:
        ms = [_make_m("jpeg", 40, 40.0, 4000), _make_m("jpeg", 80, 80.0, 8000)]
        assert interpolate_metric_at_quality(ms, "jpeg", 100.0, "ssimulacra2") is None


# ---------------------------------------------------------------------------
# compute_cross_format_cv
# ---------------------------------------------------------------------------


class TestComputeCrossFormatCV:
    """Tests for compute_cross_format_cv (coefficient-of-variation image selector)."""

    def _two_format_measurements(self) -> list[dict]:
        """jpeg bpp=0.1 and avif bpp=0.3 at ssimulacra2=70."""
        # At q=50, ssim=60; at q=80, ssim=80 → target 70 → quality≈65
        # jpeg: file_size gives bpp=0.1 at q=50, bpp=0.2 at q=80
        # avif: file_size gives bpp=0.3 at q=50, bpp=0.6 at q=80
        return [
            _make_m("jpeg", 50, 60.0, 1000),  # bpp=0.10
            _make_m("jpeg", 80, 80.0, 2000),  # bpp=0.20
            _make_m("avif", 50, 60.0, 3000),  # bpp=0.30
            _make_m("avif", 80, 80.0, 6000),  # bpp=0.60
        ]

    def test_returns_positive_cv(self) -> None:
        ms = self._two_format_measurements()
        cv = compute_cross_format_cv(
            ms, "img.png", "format", "ssimulacra2", 70.0, "bytes_per_pixel"
        )
        assert cv is not None
        assert cv > 0.0

    def test_cv_is_std_over_mean(self) -> None:
        """Verify the CV arithmetic manually."""
        ms = self._two_format_measurements()
        cv = compute_cross_format_cv(
            ms, "img.png", "format", "ssimulacra2", 70.0, "bytes_per_pixel"
        )
        # At ssim=70 (midpoint between 60 and 80) → quality=65
        # jpeg bpp at q=65: lerp(0.1,0.2) = 0.15
        # avif bpp at q=65: lerp(0.3,0.6) = 0.45
        # mean = 0.3, std = sqrt(((0.15-0.3)^2+(0.45-0.3)^2)/2) = 0.15
        # CV = 0.15/0.3 = 0.5
        assert cv == pytest.approx(0.5, rel=1e-4)

    def test_returns_none_when_only_one_format(self) -> None:
        ms = [_make_m("jpeg", 50, 60.0, 1000), _make_m("jpeg", 80, 80.0, 2000)]
        cv = compute_cross_format_cv(
            ms, "img.png", "format", "ssimulacra2", 70.0, "bytes_per_pixel"
        )
        assert cv is None

    def test_returns_none_when_target_out_of_range(self) -> None:
        ms = self._two_format_measurements()
        cv = compute_cross_format_cv(
            ms, "img.png", "format", "ssimulacra2", 99.0, "bytes_per_pixel"
        )
        assert cv is None

    def test_returns_none_when_all_outputs_are_zero(self) -> None:
        """Returns None when the mean output metric is zero (CV undefined)."""
        ms = [
            _make_m("jpeg", 50, 60.0, 0),  # bpp=0 (zero file size)
            _make_m("jpeg", 80, 80.0, 0),
            _make_m("avif", 50, 60.0, 0),
            _make_m("avif", 80, 80.0, 0),
        ]
        # bytes_per_pixel will be 0 → mean=0 → CV undefined → None
        cv = compute_cross_format_cv(
            ms, "img.png", "format", "ssimulacra2", 70.0, "bytes_per_pixel"
        )
        assert cv is None

    def test_higher_spread_gives_higher_cv(self) -> None:
        """Image with more format-spread produces higher CV than one with less spread."""
        # Low-spread image: jpeg ~0.2 bpp, avif ~0.22 bpp at target ssim
        low = [
            _make_m("jpeg", 50, 60.0, 2000, "low.png"),
            _make_m("jpeg", 80, 80.0, 4000, "low.png"),
            _make_m("avif", 50, 60.0, 2200, "low.png"),
            _make_m("avif", 80, 80.0, 4400, "low.png"),
        ]
        # High-spread image: jpeg ~0.1 bpp, avif ~0.5 bpp at target ssim
        high = [
            _make_m("jpeg", 50, 60.0, 1000, "high.png"),
            _make_m("jpeg", 80, 80.0, 2000, "high.png"),
            _make_m("avif", 50, 60.0, 5000, "high.png"),
            _make_m("avif", 80, 80.0, 10000, "high.png"),
        ]
        cv_low = compute_cross_format_cv(
            low, "low.png", "format", "ssimulacra2", 70.0, "bytes_per_pixel"
        )
        cv_high = compute_cross_format_cv(
            high, "high.png", "format", "ssimulacra2", 70.0, "bytes_per_pixel"
        )
        assert cv_low is not None
        assert cv_high is not None
        assert cv_high > cv_low


# ---------------------------------------------------------------------------
# select_best_image
# ---------------------------------------------------------------------------


class TestSelectBestImage:
    def test_selects_image_with_higher_cv(self) -> None:
        """Image with greater cross-format spread (relative to mean) is selected."""
        # low-spread image1: jpeg bpp≈0.2, avif bpp≈0.22
        # high-spread image2: jpeg bpp≈0.1, avif bpp≈0.5
        ms = [
            _make_m("jpeg", 50, 60.0, 2000, "img1.png"),
            _make_m("jpeg", 80, 80.0, 4000, "img1.png"),
            _make_m("avif", 50, 60.0, 2200, "img1.png"),
            _make_m("avif", 80, 80.0, 4400, "img1.png"),
            _make_m("jpeg", 50, 60.0, 1000, "img2.png"),
            _make_m("jpeg", 80, 80.0, 2000, "img2.png"),
            _make_m("avif", 50, 60.0, 5000, "img2.png"),
            _make_m("avif", 80, 80.0, 10000, "img2.png"),
        ]
        result = select_best_image(ms, "format", "ssimulacra2", [70.0], "bytes_per_pixel")
        assert result == "img2.png"

    def test_returns_none_when_no_valid_measurements(self) -> None:
        result = select_best_image([], "format", "ssimulacra2", [70.0], "bytes_per_pixel")
        assert result is None

    def test_excludes_images_by_basename(self) -> None:
        ms = [
            _make_m("jpeg", 50, 60.0, 1000, "img1.png"),
            _make_m("jpeg", 80, 80.0, 2000, "img1.png"),
            _make_m("avif", 50, 60.0, 5000, "img1.png"),
            _make_m("avif", 80, 80.0, 10000, "img1.png"),
            _make_m("jpeg", 50, 60.0, 2000, "img2.png"),
            _make_m("jpeg", 80, 80.0, 4000, "img2.png"),
            _make_m("avif", 50, 60.0, 2200, "img2.png"),
            _make_m("avif", 80, 80.0, 4400, "img2.png"),
        ]
        result = select_best_image(
            ms, "format", "ssimulacra2", [70.0], "bytes_per_pixel", exclude_images=["img1.png"]
        )
        assert result == "img2.png"

    def test_returns_none_when_all_excluded(self) -> None:
        ms = [
            _make_m("jpeg", 50, 60.0, 1000, "img1.png"),
            _make_m("jpeg", 80, 80.0, 2000, "img1.png"),
        ]
        result = select_best_image(
            ms, "format", "ssimulacra2", [70.0], "bytes_per_pixel", exclude_images=["img1.png"]
        )
        assert result is None
