"""Interpolation utilities for quality-metric target matching.

This module provides functions to estimate encoder quality settings that
produce a desired output metric value (e.g., SSIMULACRA2 = 70 or
bytes_per_pixel = 0.3) using linear interpolation between measured data
points from the pipeline.

The typical workflow is:

1. Load pipeline measurements from ``quality.json``.
2. For each image × format combination, gather sorted
   ``(encoder_quality, measured_metric)`` pairs.
3. Linearly interpolate the encoder quality required to hit a target
   metric value.

All interpolation is linear between the two nearest bracketing measured
points.  If the target falls outside the measured range, ``None`` is
returned.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# Quality metrics where *higher* measured value → *better* quality.
_HIGHER_IS_BETTER: set[str] = {"ssimulacra2", "psnr", "ssim"}

# Minimum number of distinct data points required to use cubic-spline
# interpolation.  With fewer points the function falls back to linear
# interpolation so that sparse measurements still produce reasonable results.
_SPLINE_MIN_POINTS: int = 4


# ---------------------------------------------------------------------------
# Core interpolation
# ---------------------------------------------------------------------------


def _lerp(x0: float, y0: float, x1: float, y1: float, y_target: float) -> float:
    """Linearly interpolate x-value at *y_target* given two (x, y) points."""
    if y1 == y0:
        return (x0 + x1) / 2.0
    return x0 + (x1 - x0) * (y_target - y0) / (y1 - y0)


def interpolate_quality_for_metric(
    measurements: list[dict[str, Any]],
    fmt: str,
    metric: str,
    target_value: float,
    *,
    source_image: str | None = None,
    speed: int | None = None,
    effort: int | None = None,
    method: int | None = None,
    chroma_subsampling: str | None = None,
    resolution: int | None = None,
) -> float | None:
    """Find the encoder quality setting that produces *target_value* of *metric*.

    Filters measurements by format (and optionally by source image and
    other encoder parameters), builds sorted ``(quality, metric_value)``
    pairs, and linearly interpolates.

    For ``bytes_per_pixel`` the metric value is computed from
    ``file_size / (width * height)``.

    Args:
        measurements: List of measurement dicts (from ``quality.json``).
        fmt: Encoder format to filter on.
        metric: Target metric name.
        target_value: Desired metric value.
        source_image: If given, restrict to this source image path.
        speed: Filter on AVIF speed.
        effort: Filter on JXL effort.
        method: Filter on WebP method.
        chroma_subsampling: Filter on chroma subsampling.
        resolution: Filter on resolution.

    Returns:
        Interpolated encoder quality as a float, or ``None`` if the
        target falls outside the measured range.
    """
    points = _collect_quality_metric_pairs(
        measurements,
        fmt,
        metric,
        source_image=source_image,
        speed=speed,
        effort=effort,
        method=method,
        chroma_subsampling=chroma_subsampling,
        resolution=resolution,
    )
    if len(points) < 2:  # noqa: PLR2004
        return None

    return _interpolate_target(points, target_value)


def interpolate_metric_at_quality(
    measurements: list[dict[str, Any]],
    fmt: str,
    quality: float,
    output_metric: str,
    *,
    source_image: str | None = None,
    speed: int | None = None,
    effort: int | None = None,
    method: int | None = None,
    chroma_subsampling: str | None = None,
    resolution: int | None = None,
) -> float | None:
    """Interpolate an output metric at a given encoder quality.

    Uses cubic-spline interpolation when at least ``_SPLINE_MIN_POINTS``
    measured quality levels are available.  Falls back to piecewise-linear
    interpolation for sparser data so that measurements with only a handful
    of quality levels still produce valid results.

    Args:
        measurements: List of measurement dicts.
        fmt: Encoder format to filter on.
        quality: The encoder quality setting (may be fractional).
        output_metric: Metric to interpolate.
        source_image: If given, restrict to this source image path.
        speed: Filter on AVIF speed.
        effort: Filter on JXL effort.
        method: Filter on WebP method.
        chroma_subsampling: Filter on chroma subsampling.
        resolution: Filter on resolution.

    Returns:
        Interpolated metric value, or ``None`` if out of range.
    """
    points = _collect_quality_metric_pairs(
        measurements,
        fmt,
        output_metric,
        source_image=source_image,
        speed=speed,
        effort=effort,
        method=method,
        chroma_subsampling=chroma_subsampling,
        resolution=resolution,
    )
    if len(points) < 2:  # noqa: PLR2004
        return None

    qs = [p[0] for p in points]
    ms = [p[1] for p in points]

    # Out-of-range check (no extrapolation)
    if quality < qs[0] or quality > qs[-1]:
        return None

    if len(points) < _SPLINE_MIN_POINTS:
        # Piecewise-linear fallback for sparse data
        for i in range(len(points) - 1):
            if qs[i] <= quality <= qs[i + 1]:
                return _lerp(ms[i], qs[i], ms[i + 1], qs[i + 1], quality)
        return None

    # Cubic-spline interpolation: metric as a function of encoder quality
    spl = CubicSpline(qs, ms, extrapolate=False)
    result = float(spl(quality))
    return None if math.isnan(result) else result


# ---------------------------------------------------------------------------
# Per-image aggregate helpers (used for image selection)
# ---------------------------------------------------------------------------


def compute_cross_format_cv(
    measurements: list[dict[str, Any]],
    source_image: str,
    tile_param: str,
    target_metric: str,
    target_value: float,
    output_metric: str,
) -> float | None:
    """Compute the coefficient of variation (CV) of *output_metric* across tile-parameter values.

    For a given source image, for each unique value of *tile_param*
    (e.g., each format), interpolates the encoder quality that achieves
    *target_value* of *target_metric*, then interpolates what
    *output_metric* would be at that quality.  Returns
    ``std / mean`` (relative standard deviation) of *output_metric*
    across tile-parameter values.

    Using CV instead of raw variance prevents bias towards images or
    fragments with inherently high absolute metric values: a low-quality
    image has the same chance of being selected as a high-quality one
    if its *relative* spread across tile-parameter values is equally
    large.

    Args:
        measurements: All measurements.
        source_image: Source image path.
        tile_param: Parameter that varies within comparison tiles (e.g. "format").
        target_metric: Metric being targeted (e.g. "ssimulacra2").
        target_value: Target value of target_metric.
        output_metric: Metric whose CV we compute (e.g. "bytes_per_pixel").

    Returns:
        CV (std / mean) of output_metric across tile-param values, or
        ``None`` if fewer than 2 tile-param values have valid
        interpolations or if the mean is zero.
    """
    # Collect unique tile-param values for this image
    img_ms = [
        m
        for m in measurements
        if m.get("source_image") == source_image or m.get("original_image") == source_image
    ]
    tile_values = sorted({str(m.get(tile_param)) for m in img_ms if m.get(tile_param) is not None})

    output_values: list[float] = []
    for tv in tile_values:
        # Filter to this tile-param value
        tile_ms = [m for m in img_ms if str(m.get(tile_param)) == tv]
        if not tile_ms:
            continue

        # Extract the format and other fixed params from the first measurement
        fmt = tile_ms[0].get("format", "")
        fixed_kwargs = _extract_fixed_params(tile_ms[0], tile_param)

        # Interpolate the quality that achieves the target
        quality = interpolate_quality_for_metric(
            tile_ms,
            fmt,
            target_metric,
            target_value,
            source_image=source_image,
            **fixed_kwargs,
        )
        if quality is None:
            continue

        # Interpolate the output metric at that quality
        out_val = interpolate_metric_at_quality(
            tile_ms,
            fmt,
            quality,
            output_metric,
            source_image=source_image,
            **fixed_kwargs,
        )
        if out_val is not None:
            output_values.append(out_val)

    if len(output_values) < 2:  # noqa: PLR2004
        return None

    mean = sum(output_values) / len(output_values)
    if mean == 0.0:
        return None
    variance = sum((v - mean) ** 2 for v in output_values) / len(output_values)
    std = math.sqrt(variance)
    return std / mean


def select_best_image(
    measurements: list[dict[str, Any]],
    tile_param: str,
    target_metric: str,
    target_values: list[float],
    output_metric: str,
    exclude_images: list[str] | None = None,
) -> str | None:
    """Select the source image with highest mean anisotropic relative standard deviation.

    For each source image and each target value, computes the
    cross-format coefficient of variation (CV = std / mean) of
    *output_metric*.  The image with the highest mean CV across all
    target values is returned.

    Using CV instead of raw variance avoids privileging high-metric
    images: a darker / lower-quality image is equally likely to be
    selected if its *relative* spread across formats is as large.

    Args:
        measurements: All measurements from quality.json.
        tile_param: Parameter that varies within comparison tiles.
        target_metric: Metric being targeted.
        target_values: List of target values.
        output_metric: Metric whose CV we maximise.
        exclude_images: Optional list of image filenames to skip.
            Matched against the basename of each image path.

    Returns:
        Path string of the selected source image, or ``None`` if
        no valid image could be found.
    """
    valid = [
        m
        for m in measurements
        if m.get("measurement_error") is None and m.get("ssimulacra2") is not None
    ]
    if not valid:
        return None

    # Collect unique source images
    images = sorted({m.get("original_image", m.get("source_image", "")) for m in valid})
    images = [img for img in images if img]

    # Apply exclusion filter
    if exclude_images:
        excluded = set(exclude_images)
        images = [img for img in images if Path(img).name not in excluded]

    best_image: str | None = None
    best_score = float("-inf")

    for img in images:
        cvs: list[float] = []
        for tv in target_values:
            v = compute_cross_format_cv(valid, img, tile_param, target_metric, tv, output_metric)
            if v is not None:
                cvs.append(v)

        if cvs:
            score = sum(cvs) / len(cvs)
            if score > best_score:
                best_score = score
                best_image = img

    return best_image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_quality_metric_pairs(
    measurements: list[dict[str, Any]],
    fmt: str,
    metric: str,
    *,
    source_image: str | None = None,
    speed: int | None = None,
    effort: int | None = None,
    method: int | None = None,
    chroma_subsampling: str | None = None,
    resolution: int | None = None,
) -> list[tuple[float, float]]:
    """Collect (quality, metric_value) pairs sorted by quality.

    Filters measurements by format and optional constraints, averages
    metric values that share the same quality setting (e.g. across
    multiple source images when source_image is None), and returns
    sorted pairs.
    """
    filtered: list[dict[str, Any]] = []
    for m in measurements:
        if m.get("measurement_error") is not None:
            continue
        if m.get("format") != fmt:
            continue
        if (
            source_image is not None
            and m.get("original_image") != source_image
            and m.get("source_image") != source_image
        ):
            continue
        if speed is not None and m.get("speed") != speed:
            continue
        if effort is not None and m.get("effort") != effort:
            continue
        if method is not None and m.get("method") != method:
            continue
        if chroma_subsampling is not None and m.get("chroma_subsampling") != chroma_subsampling:
            continue
        if resolution is not None and m.get("resolution") != resolution:
            continue

        # Compute metric value
        if metric == "bytes_per_pixel":
            w = m.get("width", 0)
            h = m.get("height", 0)
            fs = m.get("file_size", 0)
            if w > 0 and h > 0 and fs > 0:
                val = fs / (w * h)
            else:
                continue
        else:
            val = m.get(metric)
            if val is None:
                continue

        q = m.get("quality")
        if q is None:
            continue

        filtered.append({"quality": float(q), "value": float(val)})

    if not filtered:
        return []

    # Average by quality when multiple measurements exist per quality level
    from collections import defaultdict

    by_quality: dict[float, list[float]] = defaultdict(list)
    for item in filtered:
        by_quality[item["quality"]].append(item["value"])

    pairs = [(q, sum(vs) / len(vs)) for q, vs in by_quality.items()]
    pairs.sort(key=lambda p: p[0])  # Sort by quality ascending
    return pairs


def _interpolate_target(
    points: list[tuple[float, float]],
    target_value: float,
) -> float | None:
    """Interpolate quality (x) at target metric value (y).

    Points are sorted by quality ascending.  For higher-is-better
    metrics, metric values should generally increase with quality.
    For lower-is-better metrics (butteraugli), they decrease.

    When at least ``_SPLINE_MIN_POINTS`` data points are available a
    cubic spline is fitted over the full quality range and the target
    quality is located by Brent's method inside the bracketing interval.
    With fewer points the function falls back to piecewise-linear
    interpolation.
    """
    # Locate the adjacent-pair interval that brackets the target value
    bracket_idx: int | None = None
    for i in range(len(points) - 1):
        q0, m0 = points[i]
        q1, m1 = points[i + 1]
        lo, hi = min(m0, m1), max(m0, m1)
        if lo <= target_value <= hi:
            bracket_idx = i
            break

    if bracket_idx is None:
        return None

    q0, m0 = points[bracket_idx]
    q1, m1 = points[bracket_idx + 1]

    if len(points) < _SPLINE_MIN_POINTS:
        # Piecewise-linear fallback for sparse data
        return _lerp(q0, m0, q1, m1, target_value)

    # Cubic-spline interpolation: use the full dataset to build the spline,
    # then locate the target within the bracketing quality interval via
    # Brent's root-finding method.
    qs = [p[0] for p in points]
    ms = [p[1] for p in points]
    spl = CubicSpline(qs, ms)
    try:
        return float(brentq(lambda q: float(spl(q)) - target_value, q0, q1))
    except ValueError:
        # brentq requires a sign change at the interval endpoints; fall back
        # to linear if the spline overshoots within the bracket.
        return _lerp(q0, m0, q1, m1, target_value)


def _extract_fixed_params(measurement: dict[str, Any], tile_param: str) -> dict[str, Any]:
    """Extract encoder parameters from a measurement, excluding tile_param.

    Returns a dict suitable for passing as **kwargs to interpolation
    functions.
    """
    param_map = {
        "speed": "speed",
        "effort": "effort",
        "method": "method",
        "chroma_subsampling": "chroma_subsampling",
        "resolution": "resolution",
    }
    result: dict[str, Any] = {}
    for param_name, kwarg_name in param_map.items():
        if param_name == tile_param:
            continue
        val = measurement.get(param_name)
        if val is not None:
            result[kwarg_name] = val
    return result
