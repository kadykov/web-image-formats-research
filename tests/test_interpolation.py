"""Tests for the interpolation module."""

from __future__ import annotations

import pytest

from src.interpolation import (
    _SPLINE_MIN_POINTS,
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

    def test_bits_per_pixel(self) -> None:
        # 100×100 image, file_size=1000 → bpp = 0.8
        ms = [_make_m("jpeg", 50, 55.0, 1000), _make_m("jpeg", 80, 75.0, 2000)]
        pairs = _collect_quality_metric_pairs(ms, "jpeg", "bits_per_pixel")
        assert pairs[0][1] == pytest.approx(0.8)
        assert pairs[1][1] == pytest.approx(1.6)

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

    def test_filters_by_crop(self) -> None:
        m1 = {**_make_m("avif", 50, 55.0, 5000), "crop": 800}
        m2 = {**_make_m("avif", 50, 70.0, 4000), "crop": 400}
        pairs = _collect_quality_metric_pairs([m1, m2], "avif", "ssimulacra2", crop=800)
        assert len(pairs) == 1
        assert pairs[0][1] == pytest.approx(55.0)

    def test_crop_none_returns_all(self) -> None:
        m1 = {**_make_m("avif", 50, 55.0, 5000), "crop": 800}
        m2 = {**_make_m("avif", 80, 70.0, 4000), "crop": 400}
        pairs = _collect_quality_metric_pairs([m1, m2], "avif", "ssimulacra2", crop=None)
        assert len(pairs) == 2


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
        # bits-per-pixel decreases as quality setting increases (unusual but valid)
        points = [(30.0, 4.0), (80.0, 0.8)]
        # target 2.4 is between 4.0 and 0.8 → quality ~55
        result = _interpolate_target(points, 2.4)
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
# Spline vs. linear-fallback behaviour
# ---------------------------------------------------------------------------


class TestSplineInterpolation:
    """Tests for the cubic-spline code path (>= _SPLINE_MIN_POINTS data points)."""

    def _make_quadratic_ms(self, fmt: str = "jpeg") -> list[dict]:
        """Four measurements following a quadratic quality→ssimulacra2 curve.

        ssimulacra2 = (quality / 10) ** 2, so:
          q=10 → 1,  q=40 → 16,  q=70 → 49,  q=100 → 100
        """
        return [
            _make_m(fmt, 10, 1.0, 1000),
            _make_m(fmt, 40, 16.0, 4000),
            _make_m(fmt, 70, 49.0, 7000),
            _make_m(fmt, 100, 100.0, 10000),
        ]

    # --- basic sanity checks ------------------------------------------------

    def test_spline_min_points_constant_is_four(self) -> None:
        assert _SPLINE_MIN_POINTS == 4

    def test_interpolate_quality_for_metric_returns_in_range(self) -> None:
        ms = self._make_quadratic_ms()
        result = interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 25.0)
        assert result is not None
        # True solution: q = 10 * sqrt(25) = 50; must lie between 40 and 70
        assert 40.0 < result < 70.0

    def test_interpolate_metric_at_quality_returns_in_range(self) -> None:
        ms = self._make_quadratic_ms()
        result = interpolate_metric_at_quality(ms, "jpeg", 55.0, "ssimulacra2")
        assert result is not None
        assert 16.0 < result < 49.0

    # --- spline accuracy vs. piecewise-linear -------------------------------

    def test_spline_closer_to_truth_than_linear_for_nonlinear_data(self) -> None:
        """Spline estimate should be more accurate than linear for curved data."""
        ms = self._make_quadratic_ms()
        spline_result = interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 25.0)
        # True quality: q = 10 * sqrt(25) = 50.0
        true_q = 50.0
        # Piecewise-linear estimate between (40,16) and (70,49):
        #   q = 40 + (70-40) * (25-16) / (49-16) ≈ 48.18
        linear_est = 40.0 + (70.0 - 40.0) * (25.0 - 16.0) / (49.0 - 16.0)
        assert spline_result is not None
        assert abs(spline_result - true_q) < abs(linear_est - true_q)

    def test_spline_differs_noticeably_from_linear_on_curved_data(self) -> None:
        """Spline and linear should give different answers for non-linear data."""
        ms = self._make_quadratic_ms()
        spline_result = interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 25.0)
        linear_est = 40.0 + (70.0 - 40.0) * (25.0 - 16.0) / (49.0 - 16.0)
        assert spline_result is not None
        assert abs(spline_result - linear_est) > 0.5

    # --- out-of-range still returns None ------------------------------------

    def test_out_of_range_quality_for_metric_returns_none(self) -> None:
        ms = self._make_quadratic_ms()
        assert interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 200.0) is None

    def test_out_of_range_metric_at_quality_returns_none(self) -> None:
        ms = self._make_quadratic_ms()
        assert interpolate_metric_at_quality(ms, "jpeg", 110.0, "ssimulacra2") is None

    # --- linear fallback with sparse data (< _SPLINE_MIN_POINTS) -----------

    def test_two_point_linear_fallback_quality_for_metric(self) -> None:
        """With 2 data points the result must equal the linear (_lerp) answer."""
        ms = [_make_m("jpeg", 50, 50.0, 5000), _make_m("jpeg", 90, 90.0, 9000)]
        result = interpolate_quality_for_metric(ms, "jpeg", "ssimulacra2", 70.0)
        # Linear: q = 50 + (90-50) * (70-50) / (90-50) = 70
        assert result == pytest.approx(70.0)

    def test_two_point_linear_fallback_metric_at_quality(self) -> None:
        """With 2 data points the result must equal the linear (_lerp) answer."""
        ms = [_make_m("jpeg", 40, 40.0, 4000), _make_m("jpeg", 80, 80.0, 8000)]
        result = interpolate_metric_at_quality(ms, "jpeg", 60.0, "ssimulacra2")
        assert result == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# compute_cross_format_cv
# ---------------------------------------------------------------------------


class TestComputeCrossFormatCV:
    """Tests for compute_cross_format_cv (coefficient-of-variation image selector)."""

    def _two_format_measurements(self) -> list[dict]:
        """jpeg bpp=0.8 and avif bpp=2.4 at ssimulacra2=70."""
        # At q=50, ssim=60; at q=80, ssim=80 → target 70 → quality≈65
        # jpeg: file_size gives bpp=0.8 at q=50, bpp=1.6 at q=80
        # avif: file_size gives bpp=2.4 at q=50, bpp=4.8 at q=80
        return [
            _make_m("jpeg", 50, 60.0, 1000),  # bpp=0.8
            _make_m("jpeg", 80, 80.0, 2000),  # bpp=1.6
            _make_m("avif", 50, 60.0, 3000),  # bpp=2.4
            _make_m("avif", 80, 80.0, 6000),  # bpp=4.8
        ]

    def test_returns_positive_cv(self) -> None:
        ms = self._two_format_measurements()
        cv = compute_cross_format_cv(ms, "img.png", "format", "ssimulacra2", 70.0, "bits_per_pixel")
        assert cv is not None
        assert cv > 0.0

    def test_cv_is_std_over_mean(self) -> None:
        """Verify the CV arithmetic manually."""
        ms = self._two_format_measurements()
        cv = compute_cross_format_cv(ms, "img.png", "format", "ssimulacra2", 70.0, "bits_per_pixel")
        # At ssim=70 (midpoint between 60 and 80) → quality=65
        # jpeg bpp at q=65: lerp(0.8,1.6) = 1.2
        # avif bpp at q=65: lerp(2.4,4.8) = 3.6
        # mean = 2.4, std = sqrt(((1.2-2.4)^2+(3.6-2.4)^2)/2) = 1.2
        # CV = 1.2/2.4 = 0.5
        assert cv == pytest.approx(0.5, rel=1e-4)

    def test_returns_none_when_only_one_format(self) -> None:
        ms = [_make_m("jpeg", 50, 60.0, 1000), _make_m("jpeg", 80, 80.0, 2000)]
        cv = compute_cross_format_cv(ms, "img.png", "format", "ssimulacra2", 70.0, "bits_per_pixel")
        assert cv is None

    def test_returns_none_when_target_out_of_range(self) -> None:
        ms = self._two_format_measurements()
        cv = compute_cross_format_cv(ms, "img.png", "format", "ssimulacra2", 99.0, "bits_per_pixel")
        assert cv is None

    def test_returns_none_when_all_outputs_are_zero(self) -> None:
        """Returns None when the mean output metric is zero (CV undefined)."""
        ms = [
            _make_m("jpeg", 50, 60.0, 0),  # bpp=0 (zero file size)
            _make_m("jpeg", 80, 80.0, 0),
            _make_m("avif", 50, 60.0, 0),
            _make_m("avif", 80, 80.0, 0),
        ]
        # bits_per_pixel will be 0 → mean=0 → CV undefined → None
        cv = compute_cross_format_cv(ms, "img.png", "format", "ssimulacra2", 70.0, "bits_per_pixel")
        assert cv is None

    def test_higher_spread_gives_higher_cv(self) -> None:
        """Image with more format-spread produces higher CV than one with less spread."""
        # Low-spread image: jpeg ~1.6 bpp, avif ~1.76 bpp at target ssim
        low = [
            _make_m("jpeg", 50, 60.0, 2000, "low.png"),
            _make_m("jpeg", 80, 80.0, 4000, "low.png"),
            _make_m("avif", 50, 60.0, 2200, "low.png"),
            _make_m("avif", 80, 80.0, 4400, "low.png"),
        ]
        # High-spread image: jpeg ~0.8 bpp, avif ~4.0 bpp at target ssim
        high = [
            _make_m("jpeg", 50, 60.0, 1000, "high.png"),
            _make_m("jpeg", 80, 80.0, 2000, "high.png"),
            _make_m("avif", 50, 60.0, 5000, "high.png"),
            _make_m("avif", 80, 80.0, 10000, "high.png"),
        ]
        cv_low = compute_cross_format_cv(
            low, "low.png", "format", "ssimulacra2", 70.0, "bits_per_pixel"
        )
        cv_high = compute_cross_format_cv(
            high, "high.png", "format", "ssimulacra2", 70.0, "bits_per_pixel"
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
        # low-spread image1: jpeg bpp≈1.6, avif bpp≈1.76
        # high-spread image2: jpeg bpp≈0.8, avif bpp≈4.0
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
        result = select_best_image(ms, "format", "ssimulacra2", [70.0], "bits_per_pixel")
        assert result == "img2.png"

    def test_returns_none_when_no_valid_measurements(self) -> None:
        result = select_best_image([], "format", "ssimulacra2", [70.0], "bits_per_pixel")
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
            ms, "format", "ssimulacra2", [70.0], "bits_per_pixel", exclude_images=["img1.png"]
        )
        assert result == "img2.png"

    def test_returns_none_when_all_excluded(self) -> None:
        ms = [
            _make_m("jpeg", 50, 60.0, 1000, "img1.png"),
            _make_m("jpeg", 80, 80.0, 2000, "img1.png"),
        ]
        result = select_best_image(
            ms, "format", "ssimulacra2", [70.0], "bits_per_pixel", exclude_images=["img1.png"]
        )
        assert result is None
