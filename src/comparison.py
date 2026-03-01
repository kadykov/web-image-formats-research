"""Visual comparison module for identifying and visualizing encoding artifacts.

This module automates the process of finding the most representative image
fragment in a study and generating side-by-side comparison figures that show
how different encoding variants render the same region.

The workflow is:
1. Load quality measurement results
2. Compute Butteraugli distortion maps for all parameter variants of the
   selected image
3. Locate the most representative fragment using the *anisotropic variance*
   strategy (see below)
4. Crop the fragment from every variant and assemble labeled comparison grids

Fragment selection — anisotropic variance
------------------------------------------
Each comparison figure places the same fragment side-by-side encoded with
different values of the *tile parameter* (e.g. ``format`` for a multi-codec
study).  Other varying parameters (e.g. ``quality``) produce separate figures.

The most representative fragment is the one with the **highest mean
anisotropic distortion variance**: for each "split group" (a fixed
combination of the non-tile parameters), the pixel-wise variance of
distortion across all tile-parameter values is computed.  These per-group
variance maps are then averaged across all split groups to produce the
aggregate *anisotropic map*.  The sliding-window position that maximises
this map is selected as the crop region.

Fall-back: when no split group contains more than one variant (e.g. the
study sweeps only one parameter), the algorithm falls back to overall
pixel-wise variance across all variants.

When the pipeline has pre-computed fragment positions (stored in
``quality.json`` under ``"worst_fragments"``) those positions are used
directly, skipping costly on-the-fly computation.

This module requires:
- butteraugli_main (for spatial distortion maps)
- ImageMagick 7 (montage command for grid assembly)
- Pillow (for image cropping and nearest-neighbor scaling)
- Encoding tools (cjpeg, cwebp, avifenc, cjxl) matching the study formats
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

# Re-exported for backward compatibility — canonical version lives in analysis.py
from src.analysis import load_quality_results as load_quality_results
from src.encoder import ImageEncoder
from src.quality import WorstRegion, find_worst_region_in_array, read_pfm, to_png


@dataclass
class ComparisonConfig:
    """Configuration for visual comparison generation.

    Attributes:
        crop_size: Size of the crop region in original pixels (before zoom).
        zoom_factor: Factor to scale the crop (e.g., 3 for 300% zoom).
        metric: Primary metric used to find the worst measurement.
        max_columns: Maximum number of images per row in the grid.
        label_font_size: Font size for labels in the comparison grid.
        strategy: Which selection strategy to use for **both** image and
            fragment selection.  One of:

            * ``"both"`` *(default)* — run both *average* and *variance*
              strategies end-to-end, outputting separate subdirectories
              for each.
            * ``"average"`` — average the metric scores / distortion maps
              across all parameter combinations.  Identifies the image
              and fragment that are consistently most challenging.
            * ``"variance"`` — compute the variance of metric scores /
              distortion maps.  Identifies the image and fragment most
              sensitive to parameter changes.

        strategy: Fragment-selection strategy to use.  Currently only
            ``"anisotropic"`` is supported.  This field is kept for
            future extensibility.
        distmap_vmax: Upper bound of the fixed Butteraugli distortion
            scale used in the distortion-map comparison grid.  All
            per-pixel values are clamped to ``[0, distmap_vmax]`` before
            mapping to the viridis colormap, ensuring every tile uses an
            identical colour scale so structural differences between
            encoding variants are directly comparable.  Typical
            Butteraugli scores for well-encoded images fall below 2;
            scores above 10 represent severe quality loss.
            Defaults to ``5.0``.
        source_image: Optional explicit source image path (relative to
            project root) to use instead of automatic selection.
        tile_parameter: The encoding parameter that should vary within
            each comparison figure — i.e. each tile shows a different
            value of this parameter while the other varying parameters
            are held fixed per figure.  When multiple values of the
            other parameters exist they produce separate figures.

            For multi-format studies (``"format"``) quality-level
            matching is index-based: ``format_a[i]`` is compared with
            ``format_b[i]`` regardless of their absolute quality values.

            When ``None`` the value is taken from the quality.json
            metadata (propagated from the study config).  If that is
            also absent the built-in heuristic is used:
            ``"format"`` when multiple formats vary, otherwise the
            first non-quality sweep parameter.
    """

    crop_size: int = 128
    zoom_factor: int = 3
    metric: str = "ssimulacra2"
    max_columns: int = 4
    label_font_size: int = 22
    strategy: str = "anisotropic"
    distmap_vmax: float = 5.0
    source_image: str | None = None
    tile_parameter: str | None = None


@dataclass
class StrategyResult:
    """Result for an image + fragment selection run.

    Attributes:
        strategy: The strategy used.  Currently always ``"anisotropic"``;
            kept for future extensibility.
        source_image: Path (relative to project root) of the selected source image.
        image_score: The anisotropic variance score that determined image
            selection — mean within-group metric variance across all split groups.
        worst_format: Format of the single worst measurement for this image.
        worst_quality: Quality parameter of the single worst measurement.
        worst_metric_value: The single worst metric score for this image.
        region: The detected worst fragment coordinates.
        output_dir: Directory where the comparison outputs were written.
        output_images: List of generated comparison image paths.
    """

    strategy: str
    source_image: str
    image_score: float
    worst_format: str
    worst_quality: int
    worst_metric_value: float
    region: WorstRegion
    output_dir: Path
    output_images: list[Path] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of the visual comparison generation.

    Attributes:
        study_id: Study identifier.
        strategies: List with one :class:`StrategyResult` entry (the
            anisotropic selection result).  Kept as a list for future
            extensibility.
        varying_parameters: Parameters that vary across measurements.
    """

    study_id: str
    strategies: list[StrategyResult] = field(default_factory=list)
    varying_parameters: list[str] = field(default_factory=list)


def find_worst_measurement(
    measurements: list[dict],
    metric: str = "ssimulacra2",
) -> dict:
    """Find the measurement with the worst quality score.

    For metrics where higher is better (SSIMULACRA2, PSNR, SSIM),
    returns the measurement with the lowest score. For metrics where
    lower is better (Butteraugli), returns the highest score.

    Only considers measurements without errors and with valid metric values.

    Args:
        measurements: List of measurement dictionaries
        metric: Metric to use for finding the worst case

    Returns:
        The measurement dictionary with the worst quality

    Raises:
        ValueError: If no valid measurements exist for the given metric
    """
    # Metrics where higher is better
    higher_is_better = {"ssimulacra2", "psnr", "ssim"}

    valid = [
        m for m in measurements if m.get("measurement_error") is None and m.get(metric) is not None
    ]

    if not valid:
        msg = f"No valid measurements found for metric '{metric}'"
        raise ValueError(msg)

    if metric in higher_is_better:
        return min(valid, key=lambda m: m[metric])
    else:
        return max(valid, key=lambda m: m[metric])


def _group_scores_by_image(
    measurements: list[dict],
    metric: str,
) -> dict[str, list[float]]:
    """Group valid metric scores by source image.

    Args:
        measurements: List of measurement dictionaries.
        metric: Metric name to extract.

    Returns:
        Mapping from source image path to list of metric values.

    Raises:
        ValueError: If no valid measurements exist.
    """
    image_scores: dict[str, list[float]] = {}
    for m in measurements:
        if m.get("measurement_error") is None and m.get(metric) is not None:
            source = m["source_image"]
            if source not in image_scores:
                image_scores[source] = []
            image_scores[source].append(m[metric])

    if not image_scores:
        msg = f"No valid measurements found for metric '{metric}'"
        raise ValueError(msg)

    return image_scores


def _anisotropic_image_score(
    measurements: list[dict],
    metric: str,
    split_params: list[str],
) -> float:
    """Compute the anisotropic variance score for a single image's measurements.

    Groups measurements by *split_params* (one group per comparison figure),
    computes the metric variance within each group across tile-parameter
    values, then returns the mean of those per-group variances.

    Falls back to overall metric variance when no split group contains
    more than one measurement (i.e. the study has only a single tile-param
    value per group).

    Args:
        measurements: Measurements for one source image (already filtered to
            valid, non-error rows with the metric present).
        metric: Metric name to score on.
        split_params: Parameters that produce separate figures (all remaining
            varying params that are *not* the tile parameter).

    Returns:
        Anisotropic variance score \u226510.  Higher means the image is more
        sensitive to tile-parameter changes.
    """
    # Group by combination of split_params values
    groups: dict[tuple, list[float]] = {}
    for m in measurements:
        key = tuple(str(m.get(p)) for p in split_params)
        groups.setdefault(key, []).append(float(m[metric]))

    group_vars: list[float] = []
    for scores in groups.values():
        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
            group_vars.append(sum((s - mean) ** 2 for s in scores) / len(scores))

    if group_vars:
        return sum(group_vars) / len(group_vars)

    # Fallback: overall variance when no group has ≥2 variants
    all_scores = [m[metric] for m in measurements]
    if len(all_scores) >= 2:
        fallback_mean: float = sum(float(s) for s in all_scores) / len(all_scores)
        return sum((float(s) - fallback_mean) ** 2 for s in all_scores) / len(all_scores)
    return 0.0


def find_worst_source_image(
    measurements: list[dict],
    metric: str = "ssimulacra2",
    tile_param: str | None = None,  # noqa: ARG001
    split_params: list[str] | None = None,
) -> str:
    """Find the source image most sensitive to tile-parameter variation.

    For each source image, groups its measurements by *split_params* (one
    group per comparison figure), computes the intra-group metric variance
    across tile-parameter values, then averages those per-group variances.
    The image with the highest mean anisotropic variance is returned.

    Falls back to overall metric variance when no split group contains
    more than one measurement (e.g. a single-format study).

    Args:
        measurements: List of measurement dictionaries.
        metric: Metric to use for comparison.
        tile_param: Parameter that varies across tiles within one figure.
            Unused for this function (the grouping key is *split_params*)
            but accepted for API symmetry.
        split_params: Parameters whose distinct values produce separate
            comparison figures.  Defaults to ``[]`` (single figure).

    Returns:
        Path string of the selected source image.

    Raises:
        ValueError: If no valid measurements exist.
    """
    if split_params is None:
        split_params = []

    valid = [
        m for m in measurements if m.get("measurement_error") is None and m.get(metric) is not None
    ]
    if not valid:
        msg = f"No valid measurements found for metric '{metric}'"
        raise ValueError(msg)

    # Collect unique source images
    source_images = sorted({m["source_image"] for m in valid})

    image_scores: dict[str, float] = {}
    for src in source_images:
        src_ms = [m for m in valid if m["source_image"] == src]
        image_scores[src] = _anisotropic_image_score(src_ms, metric, split_params)

    return max(image_scores, key=lambda k: image_scores[k])


def _find_worst_original_image(
    measurements: list[dict],
    metric: str = "ssimulacra2",
    tile_param: str | None = None,  # noqa: ARG001
    split_params: list[str] | None = None,
) -> str:
    """Find the original image most sensitive to tile-parameter variation.

    Like :func:`find_worst_source_image` but groups by ``original_image``
    instead of ``source_image``.  Used for resolution studies where each
    ``(image, resolution)`` pair has its own ``source_image`` but all share
    the same ``original_image``.

    Args:
        measurements: List of measurement dictionaries.
        metric: Metric to use for comparison.
        tile_param: Passed through for API symmetry; not used in grouping.
        split_params: Parameters that produce separate comparison figures.
            Defaults to ``[]``.

    Returns:
        The ``original_image`` path string of the selected image.

    Raises:
        ValueError: If no valid measurements exist.
    """
    if split_params is None:
        split_params = []

    valid = [
        m for m in measurements if m.get("measurement_error") is None and m.get(metric) is not None
    ]
    if not valid:
        msg = f"No valid measurements with metric '{metric}' found"
        raise ValueError(msg)

    original_images = sorted({m.get("original_image", m["source_image"]) for m in valid})

    image_scores: dict[str, float] = {}
    for orig in original_images:
        orig_ms = [m for m in valid if m.get("original_image", m["source_image"]) == orig]
        image_scores[orig] = _anisotropic_image_score(orig_ms, metric, split_params)

    return max(image_scores, key=lambda k: image_scores[k])


def get_worst_image_score(
    measurements: list[dict],
    source_image: str,
    metric: str = "ssimulacra2",
    tile_param: str | None = None,  # noqa: ARG001
    split_params: list[str] | None = None,
) -> float:
    """Compute the anisotropic variance score for a given source image.

    Args:
        measurements: List of measurement dictionaries.
        source_image: The source image path to compute the score for.
        metric: Metric name.
        tile_param: Accepted for API symmetry; not used in grouping.
        split_params: Parameters that produce separate comparison figures.
            Defaults to ``[]``.

    Returns:
        The anisotropic variance score for this image.
    """
    if split_params is None:
        split_params = []
    valid = [
        m
        for m in measurements
        if m["source_image"] == source_image
        and m.get("measurement_error") is None
        and m.get(metric) is not None
    ]
    if not valid:
        return 0.0
    return _anisotropic_image_score(valid, metric, split_params)


def generate_distortion_map(
    original: Path,
    compressed: Path,
    output_pfm: Path,
) -> Path:
    """Generate a raw Butteraugli distortion map as a PFM file.

    Uses ``butteraugli_main --rawdistmap`` to produce a PFM file
    containing the actual per-pixel float distortion values.  The false-
    colour PNG normally written by ``--distmap`` is redirected to a
    throw-away file inside a temporary directory and discarded.

    Args:
        original: Path to the original reference image.
        compressed: Path to the compressed/encoded image.
        output_pfm: Path where the raw PFM distortion map will be written.

    Returns:
        Path to the generated PFM file (same as ``output_pfm``).

    Raises:
        RuntimeError: If butteraugli_main fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Convert to PNG if needed
        orig_png = original
        comp_png = compressed
        if original.suffix.lower() != ".png":
            orig_png = tmpdir_path / "original.png"
            to_png(original, orig_png)
        if compressed.suffix.lower() != ".png":
            comp_png = tmpdir_path / "compressed.png"
            to_png(compressed, comp_png)

        output_pfm.parent.mkdir(parents=True, exist_ok=True)
        # butteraugli_main requires at least --distmap; route it to a
        # throw-away file inside the temp directory.
        cmd = [
            "butteraugli_main",
            str(orig_png),
            str(comp_png),
            "--distmap",
            str(tmpdir_path / "distmap_unused.png"),
            "--rawdistmap",
            str(output_pfm),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"butteraugli_main failed: {result.stderr}"
            raise RuntimeError(msg)

    return output_pfm


def find_worst_region(
    pfm_path: Path,
    crop_size: int = 128,
) -> WorstRegion:
    """Find the region with the highest distortion in a raw Butteraugli PFM file.

    Reads the PFM via :func:`src.quality.read_pfm` and delegates the
    sliding-window search to :func:`src.quality.find_worst_region_in_array`.

    Args:
        pfm_path: Path to a ``.pfm`` raw distortion map.
        crop_size: Side length of the square sliding window in pixels.

    Returns:
        :class:`WorstRegion` with coordinates and average distortion score.
    """
    arr = read_pfm(pfm_path)
    return find_worst_region_in_array(arr, crop_size)


def crop_and_zoom(
    image_path: Path,
    region: WorstRegion,
    zoom_factor: int = 3,
    output_path: Path | None = None,
) -> Image.Image:
    """Crop a region from an image and zoom with nearest-neighbor interpolation.

    Args:
        image_path: Path to the image file
        region: Region to crop
        zoom_factor: Scale factor (e.g., 3 for 300%)
        output_path: Optional path to save the result

    Returns:
        PIL Image of the cropped and zoomed region
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert to PNG if needed for formats Pillow can't read
        src = image_path
        if image_path.suffix.lower() in (".jxl", ".jpegxl"):
            src = Path(tmpdir) / "decoded.png"
            to_png(image_path, src)
        elif image_path.suffix.lower() == ".avif":
            try:
                src = Path(tmpdir) / "decoded.png"
                to_png(image_path, src)
            except OSError:
                src = image_path  # Try Pillow directly

        with Image.open(src) as img:
            # Clamp region to image bounds
            x = max(0, min(region.x, img.width - 1))
            y = max(0, min(region.y, img.height - 1))
            x2 = min(x + region.width, img.width)
            y2 = min(y + region.height, img.height)

            cropped = img.crop((x, y, x2, y2))

            # Zoom with nearest-neighbor (no interpolation)
            new_w = cropped.width * zoom_factor
            new_h = cropped.height * zoom_factor
            zoomed = cropped.resize((new_w, new_h), Image.Resampling.NEAREST)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        zoomed.save(output_path, format="PNG")

    return zoomed


def determine_varying_parameters(measurements: list[dict]) -> list[str]:
    """Determine which encoding parameters vary across measurements.

    Args:
        measurements: List of measurement dictionaries

    Returns:
        List of parameter names that have more than one unique value
    """
    candidates = [
        "format",
        "quality",
        "chroma_subsampling",
        "speed",
        "effort",
        "method",
        "resolution",
    ]
    varying = []
    for param in candidates:
        values = {m.get(param) for m in measurements if m.get(param) is not None}
        if len(values) > 1:
            varying.append(param)
    return varying


def _build_label(measurement: dict, varying: list[str]) -> str:
    """Build a descriptive label for a measurement tile.

    Includes format and all varying parameters in the label.

    Args:
        measurement: Measurement dictionary
        varying: List of varying parameter names

    Returns:
        Label string like "AVIF q50 speed=3"
    """
    parts = [measurement["format"].upper()]
    for param in varying:
        value = measurement.get(param)
        if value is not None:
            if param == "quality":
                parts.append(f"q{value}")
            elif param == "chroma_subsampling":
                parts.append(f"{value}")
            elif param == "format":
                continue  # Already included
            else:
                parts.append(f"{param}={value}")
    return " ".join(parts)


def _build_metric_label(measurement: dict) -> str:
    """Build a label showing key metric values.

    Args:
        measurement: Measurement dictionary

    Returns:
        Label string like "SSIM2: 75.5 | Butteraugli: 2.5"
    """
    parts = []
    if measurement.get("ssimulacra2") is not None:
        parts.append(f"SSIM2:{measurement['ssimulacra2']:.1f}")
    if measurement.get("butteraugli") is not None:
        parts.append(f"BA:{measurement['butteraugli']:.2f}")
    if measurement.get("file_size") is not None:
        size_kb = measurement["file_size"] / 1024
        parts.append(f"{size_kb:.0f}KB")
    return " ".join(parts)


def assemble_comparison_grid(
    crops: list[tuple[Path, str, str]],
    output_path: Path,
    max_columns: int = 4,
    label_font_size: int = 22,
) -> Path:
    """Assemble cropped images into a labeled grid using ImageMagick montage.

    Each image is annotated with a title label (encoding parameters) and
    a subtitle (metric values).

    Args:
        crops: List of (image_path, title_label, metric_label) tuples
        output_path: Path where the grid image will be saved
        max_columns: Maximum images per row
        label_font_size: Font size for labels

    Returns:
        Path to the generated comparison grid image

    Raises:
        RuntimeError: If montage command fails
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine grid geometry
    n = len(crops)
    cols = min(n, max_columns)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create labeled tiles using ImageMagick
        labeled_paths: list[str] = []
        for i, (img_path, title, metrics) in enumerate(crops):
            labeled_path = tmpdir_path / f"labeled_{i:03d}.png"

            # Use ImageMagick to add labels above and below the image
            combined_label = f"{title}\\n{metrics}"
            cmd = [
                "magick",
                str(img_path),
                "-gravity",
                "North",
                "-background",
                "white",
                "-splice",
                f"0x{label_font_size * 3}",
                "-font",
                "DejaVu-Sans",
                "-pointsize",
                str(label_font_size),
                "-gravity",
                "North",
                "-annotate",
                "+0+2",
                combined_label,
                str(labeled_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                msg = f"ImageMagick label failed for {img_path}: {result.stderr}"
                raise RuntimeError(msg)
            labeled_paths.append(str(labeled_path))

        # Use montage to create the grid
        lossless_args = (
            ["-define", "webp:lossless=true"] if output_path.suffix.lower() == ".webp" else []
        )
        cmd = [
            "montage",
            *labeled_paths,
            "-tile",
            f"{cols}x",
            "-geometry",
            "+4+4",
            "-background",
            "white",
            "-font",
            "DejaVu-Sans",
            *lossless_args,
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"montage failed: {result.stderr}"
            raise RuntimeError(msg)

    return output_path


def encode_image(
    source_path: Path,
    measurement: dict,
    output_dir: Path,
) -> Path | None:
    """Re-encode a source image using the parameters from a measurement record.

    Uses the same encoder tools (cjpeg, cwebp, avifenc, cjxl) as the
    original study to reproduce the encoded image on the fly.

    When the measurement includes a ``resolution`` parameter, the source
    image is resized to that resolution (longest edge) before encoding.

    Args:
        source_path: Path to the source image (PNG)
        measurement: Measurement dictionary with format, quality, and
            optional encoder parameters (speed, effort, method, chroma_subsampling)
        output_dir: Directory where the encoded file will be written

    Returns:
        Path to the encoded file, or None if encoding failed
    """
    from src.preprocessing import ImagePreprocessor

    # Handle resolution: resize source before encoding if needed
    resolution = measurement.get("resolution")
    actual_source = source_path
    _tmp_dir = None
    if resolution is not None:
        import tempfile as _tempfile

        _tmp_dir = _tempfile.mkdtemp(prefix="comparison_resize_")
        preprocessor = ImagePreprocessor(Path(_tmp_dir))
        output_name = f"{source_path.stem}_r{resolution}.png"
        actual_source = preprocessor.resize_image(
            source_path,
            target_size=(resolution, resolution),
            output_name=output_name,
            keep_aspect_ratio=True,
        )

    encoder = ImageEncoder(output_dir)
    fmt = measurement["format"]
    quality = measurement["quality"]
    output_name = f"encoded_{fmt}_q{quality}"

    # Add extra params to output name for uniqueness
    for param in ("chroma_subsampling", "speed", "effort", "method", "resolution"):
        if measurement.get(param) is not None:
            output_name += f"_{param}{measurement[param]}"

    if fmt == "jpeg":
        result = encoder.encode_jpeg(actual_source, quality, output_name=output_name)
    elif fmt == "webp":
        method = measurement.get("method") or 4
        result = encoder.encode_webp(actual_source, quality, method=method, output_name=output_name)
    elif fmt == "avif":
        speed = measurement.get("speed") or 4
        chroma = measurement.get("chroma_subsampling")
        result = encoder.encode_avif(
            actual_source,
            quality,
            speed=speed,
            chroma_subsampling=chroma,
            output_name=output_name,
        )
    elif fmt == "jxl":
        effort = measurement.get("effort") or 7
        result = encoder.encode_jxl(actual_source, quality, effort=effort, output_name=output_name)
    else:
        print(f"  Warning: Unknown format '{fmt}', skipping")
        if _tmp_dir is not None:
            import shutil as _shutil

            _shutil.rmtree(_tmp_dir, ignore_errors=True)
        return None

    if _tmp_dir is not None:
        import shutil as _shutil

        _shutil.rmtree(_tmp_dir, ignore_errors=True)

    if result.success and result.output_path is not None:
        return result.output_path
    else:
        print(f"  Warning: encoding failed for {fmt} q{quality}: {result.error_message}")
        return None


def _resolve_encoded_path(
    measurement: dict,
    project_root: Path,
) -> Path | None:
    """Check if a pre-existing encoded file is available for a measurement.

    When the pipeline was run with ``save_worst_image=True`` or
    ``save_artifacts=True``, encoded files are stored on disk and their
    relative paths are recorded in the ``encoded_path`` field of each
    measurement.

    Args:
        measurement: Measurement dictionary (from quality.json)
        project_root: Project root directory for resolving relative paths

    Returns:
        Absolute path to the encoded file if it exists, ``None`` otherwise
    """
    encoded_path: str = measurement.get("encoded_path", "")
    if encoded_path:
        full_path = project_root / encoded_path
        if full_path.exists():
            return full_path
    return None


def _get_or_encode(
    source_path: Path,
    measurement: dict,
    output_dir: Path,
    project_root: Path,
) -> Path | None:
    """Get a pre-existing encoded file or re-encode on the fly.

    Checks for saved artifacts first (from ``save_worst_image`` or
    ``save_artifacts`` pipeline options).  Falls back to re-encoding
    using the same encoder parameters as the original measurement.

    When the measurement has a ``resolution`` parameter, the source image
    is resized before encoding (handled by :func:`encode_image`).

    Args:
        source_path: Path to the original source image (before any resizing)
        measurement: Measurement dictionary
        output_dir: Directory for re-encoded files (used as fallback)
        project_root: Project root for resolving relative paths

    Returns:
        Path to the encoded file, or ``None`` if encoding failed
    """
    existing = _resolve_encoded_path(measurement, project_root)
    if existing is not None:
        return existing
    return encode_image(source_path, measurement, output_dir)


def _resolve_source_for_resolution(
    original_path: Path,
    resolution: int | None,
    cache: dict[int, Path],
    tmpdir: Path,
) -> Path:
    """Get the correct source image for a given resolution.

    For ``resolution=None`` returns the original image.  For a specific
    resolution value, resizes the original to fit within a
    ``(resolution, resolution)`` bounding box (preserving aspect ratio),
    caching the result so each resolution is only computed once.

    Args:
        original_path: Path to the original dataset image.
        resolution: Target longest-edge in pixels, or ``None`` for original.
        cache: Mutable dict mapping resolution → preprocessed path.
        tmpdir: Temporary directory for resized copies.

    Returns:
        Path to the (possibly resized) source image.
    """
    if resolution is None:
        return original_path
    if resolution in cache:
        return cache[resolution]
    from src.preprocessing import ImagePreprocessor

    res_dir = tmpdir / f"r{resolution}"
    preprocessor = ImagePreprocessor(res_dir)
    output_name = f"{original_path.stem}_r{resolution}.png"
    resized = preprocessor.resize_image(
        original_path,
        target_size=(resolution, resolution),
        output_name=output_name,
        keep_aspect_ratio=True,
    )
    cache[resolution] = resized
    return resized


def _anisotropic_variance_map(
    variant_pairs: list[tuple[np.ndarray, dict]],
    split_params: list[str],
) -> np.ndarray:
    """Compute the anisotropic variance distortion map.

    Groups variant distortion arrays by *split_params* (one group per
    comparison figure), computes pixel-wise variance within each group
    across the tile-parameter values, then averages these per-group
    variance maps.

    Falls back to overall pixel-wise variance when no split group
    contains ≥2 variants (i.e. the study sweeps only a single parameter
    and all variants go on one figure).

    Args:
        variant_pairs: List of ``(distortion_array, measurement_dict)``
            for every encoding variant that was successfully measured.
        split_params: Parameters whose distinct value combinations
            define the groups.

    Returns:
        2-D ``float64`` array of shape ``(H, W)`` with anisotropic
        variance distortion values.  Higher values indicate regions
        where the tile-parameter choice matters most.
    """
    # Group arrays by split_params key
    groups: dict[tuple, list[np.ndarray]] = {}
    for arr, m in variant_pairs:
        key = tuple(str(m.get(p)) for p in split_params)
        groups.setdefault(key, []).append(arr)

    # Compute within-group pixel-wise variance for groups with ≥2 variants
    group_var_maps: list[np.ndarray] = []
    for arrs in groups.values():
        if len(arrs) >= 2:
            stacked = np.stack(arrs, axis=0)  # (K, H, W)
            group_var_maps.append(stacked.var(axis=0))

    if group_var_maps:
        return np.stack(group_var_maps, axis=0).mean(axis=0)  # type: ignore[no-any-return]

    # Fallback: overall variance (when single-tile_param study)
    all_arrs = [arr for arr, _ in variant_pairs]
    if len(all_arrs) >= 2:
        stacked = np.stack(all_arrs, axis=0)
        return stacked.var(axis=0)  # type: ignore[no-any-return]

    # Single variant: return the distortion map itself
    return variant_pairs[0][0].astype(np.float64)


def compute_aggregate_distortion_maps(
    source_path: Path,
    measurements: list[dict],
    output_dir: Path,
    project_root: Path,
    encoded_dir: Path,
    tile_param: str | None = None,  # noqa: ARG001
    split_params: list[str] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, dict]]]:
    """Compute the anisotropic variance distortion map across all variants.

    For each measurement, obtains the encoded image (from saved artifacts
    or re-encodes on the fly), runs ``butteraugli_main --rawdistmap`` to
    get per-pixel float distortion values, then computes the *anisotropic
    variance map* using :func:`_anisotropic_variance_map`.

    The anisotropic map concentrates on regions where the *tile parameter*
    (e.g. ``format``) matters most.  Measurements are grouped by
    *split_params* (the parameters that produce separate figures); within
    each group, the pixel-wise variance is computed across all tile-param
    values.  The per-group variance maps are then averaged.

    Args:
        source_path: Path to the source (original) image.
        measurements: List of measurement dicts for the source image.
        output_dir: Directory where per-variant distortion maps will be
            written (under a ``pfms/`` subdirectory).
        project_root: Project root for resolving saved artifact paths.
        encoded_dir: Directory for re-encoded files produced on the fly.
        tile_param: Parameter varying within each comparison figure.
            Used only as context; the anisotropic grouping key is
            *split_params*.
        split_params: Parameters that produce separate comparison figures.
            Defaults to ``[]``.

    Returns:
        ``(aniso_map, variant_pairs)`` — a ``float64`` array of shape
        ``(H, W)`` and the list of ``(array, measurement)`` pairs used
        to compute it.

    Raises:
        RuntimeError: If no distortion maps could be produced for any variant.
    """
    if split_params is None:
        split_params = []

    pfms_dir = output_dir / "pfms"
    pfms_dir.mkdir(parents=True, exist_ok=True)

    variant_pairs: list[tuple[np.ndarray, dict]] = []

    for m in measurements:
        fmt = m["format"]
        quality = m["quality"]
        suffix = f"{fmt}_q{quality}"
        for param in ("chroma_subsampling", "speed", "effort", "method"):
            if m.get(param) is not None:
                suffix += f"_{param}{m[param]}"

        enc_path = _get_or_encode(source_path, m, encoded_dir, project_root)
        if enc_path is None:
            print(f"    Warning: could not obtain encoded image for {fmt} q{quality}")
            continue

        # Check for a PFM saved by the pipeline next to the encoded file
        colocated_pfm = enc_path.with_suffix(".pfm")
        if colocated_pfm.exists():
            try:
                arr = read_pfm(colocated_pfm)
                variant_pairs.append((arr, m))
                continue
            except ValueError:
                pass  # fall through to recomputing

        raw_pfm = pfms_dir / f"distmap_{suffix}.pfm"
        try:
            generate_distortion_map(source_path, enc_path, raw_pfm)
        except RuntimeError as exc:
            print(f"    Warning: distortion map failed for {fmt} q{quality}: {exc}")
            continue

        try:
            arr = read_pfm(raw_pfm)
        except ValueError as exc:
            print(f"    Warning: could not read PFM for {fmt} q{quality}: {exc}")
            continue

        variant_pairs.append((arr, m))

    if not variant_pairs:
        msg = "No distortion maps could be computed for any variant"
        raise RuntimeError(msg)

    aniso_map = _anisotropic_variance_map(variant_pairs, split_params)
    return aniso_map, variant_pairs


def _default_tile_parameter(varying: list[str]) -> str | None:
    """Determine the default tile parameter from the set of varying parameters.

    Heuristic (in priority order):

    1. If ``"format"`` varies → ``"format"`` (cross-encoder studies).
    2. If any non-quality parameter varies → the first such parameter.
    3. Otherwise → ``"quality"``.

    Returns ``None`` when ``varying`` is empty.
    """
    if not varying:
        return None
    if "format" in varying:
        return "format"
    non_quality = [p for p in varying if p != "quality"]
    if non_quality:
        return non_quality[0]
    return "quality"


def _compute_quality_indices(measurements: list[dict]) -> dict[tuple[str, int], int]:
    """Build a ``(format, quality_value) → quality_index`` mapping.

    For each format, the unique quality values found in *measurements* are
    sorted ascending and assigned 0-based indices.  This lets the comparison
    logic match ``format_a[i]`` with ``format_b[i]`` even when their quality
    scales differ (e.g. AVIF [40…90] vs JPEG [50…100]).
    """
    fmt_qualities: dict[str, set[int]] = {}
    for m in measurements:
        fmt = m.get("format") or ""
        q = m.get("quality") or 0
        fmt_qualities.setdefault(fmt, set()).add(q)

    result: dict[tuple[str, int], int] = {}
    for fmt, qs in fmt_qualities.items():
        for idx, q in enumerate(sorted(qs)):
            result[(fmt, q)] = idx
    return result


def _group_measurements_for_comparison(
    measurements: list[dict],
    tile_param: str,
    split_params: list[str],
) -> list[tuple[str, list[int]]]:
    """Group measurements into per-figure sets for comparison grid assembly.

    Returns a sorted list of ``(group_label, [indices_into_measurements])``
    where each index refers to a position in *measurements*.

    When *tile_param* is ``"format"`` and ``"quality"`` is a split parameter,
    quality matching is index-based (see :func:`_compute_quality_indices`) so
    that ``format_a[i]`` is grouped with ``format_b[i]`` regardless of their
    absolute quality values.  Formats with fewer quality levels simply produce
    fewer tiles in their respective figures.

    For all other cases grouping is done by the exact values of *split_params*.
    """
    use_quality_index = tile_param == "format" and "quality" in split_params

    if use_quality_index:
        q_indices = _compute_quality_indices(measurements)
        groups: dict[tuple, list[int]] = {}
        for i, m in enumerate(measurements):
            fmt = m.get("format") or ""
            q = m.get("quality") or 0
            q_idx = q_indices.get((fmt, q), 0)
            other = tuple(str(m.get(p)) for p in split_params if p != "quality")
            key = (q_idx,) + other
            groups.setdefault(key, []).append(i)

        return [(f"qi{key[0]}", idxs) for key, idxs in sorted(groups.items())]

    # Standard: group by the exact values of split_params
    groups_std: dict[tuple, list[int]] = {}
    for i, m in enumerate(measurements):
        key = tuple(str(m.get(p)) for p in split_params)
        groups_std.setdefault(key, []).append(i)

    result: list[tuple[str, list[int]]] = []
    for key, idxs in sorted(groups_std.items()):
        if len(split_params) == 1:
            label = f"{split_params[0]}_{key[0]}"
        elif split_params:
            label = "_".join(f"{p}-{v}" for p, v in zip(split_params, key, strict=True))
        else:
            label = "all"
        result.append((label, idxs))
    return result


def generate_comparison(
    quality_json_path: Path,
    output_dir: Path,
    project_root: Path,
    config: ComparisonConfig | None = None,
) -> ComparisonResult:
    """Generate visual comparison images for a study.

    Uses the *anisotropic variance* strategy to select the most
    representative image and fragment:

    1. Select the source image whose metric scores vary most across
       tile-parameter values (averaged over split-parameter groups).
    2. Compute Butteraugli distortion maps for every encoding variant of
       that image.
    3. Locate the fragment with the highest mean anisotropic distortion
       variance using :func:`_anisotropic_variance_map`.
    4. Crop the fragment from every variant and assemble labeled comparison
       grids.

    When the study includes a ``resolution`` sweep, steps 2–4 are
    repeated for each resolution level; outputs are placed in per-resolution
    subdirectories (e.g. ``<output_dir>/r720/``).

    When the pipeline was run with ``save_worst_image=True``, encoded
    files and pre-computed fragment positions are used directly from
    quality.json, skipping costly on-the-fly computation.

    Args:
        quality_json_path: Path to the quality.json results file.
        output_dir: Directory where comparison images will be saved.
        project_root: Project root directory for resolving relative paths.
        config: Comparison configuration (uses defaults if ``None``).

    Returns:
        :class:`ComparisonResult` with one :class:`StrategyResult`.

    Raises:
        FileNotFoundError: If quality results or images are not found.
        RuntimeError: If image processing tools fail.
        ValueError: If an unsupported strategy is requested.
    """
    if config is None:
        config = ComparisonConfig()

    if config.strategy != "anisotropic":
        msg = f"Unknown strategy {config.strategy!r}. Only 'anisotropic' is supported."
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load quality results
    data = load_quality_results(quality_json_path)
    measurements = data["measurements"]
    study_id = data.get("study_id", "unknown")
    worst_images = data.get("worst_images")
    worst_fragments = data.get("worst_fragments")

    print(f"Generating visual comparison for study: {study_id}")
    print(f"  Total measurements: {len(measurements)}")
    print("  Strategy: anisotropic")

    # Determine varying parameters
    varying = determine_varying_parameters(measurements)

    # Detect whether resolution varies — if so, we must process per-resolution
    resolution_varies = "resolution" in varying
    if resolution_varies:
        unique_resolutions: list[int | None] = sorted(
            {m.get("resolution") for m in measurements if m.get("resolution") is not None}
        )
        print(
            f"  Resolution varies: {len(unique_resolutions)} levels "
            f"({', '.join(f'r{r}' for r in unique_resolutions if r is not None)})"
        )
        intra_res_varying = [p for p in varying if p != "resolution"]
    else:
        unique_resolutions = [None]
        intra_res_varying = varying

    # Determine tile_parameter and split_params.
    # Priority: explicit CLI/config → quality.json metadata → built-in heuristic.
    tile_param: str | None = config.tile_parameter
    if tile_param is None:
        tile_param = data.get("comparison_tile_parameter")
    if tile_param is None:
        tile_param = _default_tile_parameter(intra_res_varying)
    split_params = [p for p in intra_res_varying if p != tile_param]

    print(f"  Tile parameter: {tile_param!r}")
    if split_params:
        print(f"  Split parameters (one figure per value): {split_params}")
    else:
        print("  Single comparison figure (no split parameters)")

    # Select the source image.
    # Priority: explicit override → pipeline pre-computed → on-the-fly selection.
    if config.source_image is not None:
        img_key = config.source_image
        print(f"  Using explicitly specified source image: {img_key}")
    elif worst_images and "anisotropic" in worst_images:
        img_key = worst_images["anisotropic"]["original_image"]
        print("  Using pipeline-selected worst image (anisotropic)")
    else:
        if resolution_varies:
            img_key = _find_worst_original_image(
                measurements,
                metric=config.metric,
                split_params=split_params,
            )
        else:
            img_key = find_worst_source_image(
                measurements,
                metric=config.metric,
                split_params=split_params,
            )
    print(f"  Selected image: {img_key}")

    strategy_results: list[StrategyResult] = []

    with tempfile.TemporaryDirectory() as _work:
        work = Path(_work)
        preprocess_cache: dict[int, Path] = {}

        original_path = project_root / img_key
        if not original_path.exists():
            msg = f"Source image not found: {original_path}"
            raise FileNotFoundError(msg)

        all_output_images: list[Path] = []
        first_region: WorstRegion | None = None
        first_worst_m: dict | None = None
        first_image_score: float | None = None

        for resolution in unique_resolutions:
            source_path = _resolve_source_for_resolution(
                original_path,
                resolution,
                preprocess_cache,
                work / "prep",
            )

            # Filter measurements for this image and resolution
            if resolution_varies:
                source_measurements = [
                    m
                    for m in measurements
                    if m["original_image"] == img_key
                    and m.get("resolution") == resolution
                    and m.get("measurement_error") is None
                    and m.get(config.metric) is not None
                ]
            else:
                source_measurements = [
                    m
                    for m in measurements
                    if m["source_image"] == img_key
                    and m.get("measurement_error") is None
                    and m.get(config.metric) is not None
                ]

            if not source_measurements:
                continue

            res_label = f"r{resolution}" if resolution is not None else None
            res_dir = output_dir / res_label if res_label else output_dir

            encoded_dir = work / "encoded" / (res_label or "original") / Path(img_key).stem
            encoded_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"  {f'[{res_label}] ' if res_label else ''}"
                f"Computing distortion maps for {Path(img_key).name} "
                f"({len(source_measurements)} variants)..."
            )

            aniso_map, variant_pairs = compute_aggregate_distortion_maps(
                source_path,
                source_measurements,
                work / "distmaps" / (res_label or "original") / Path(img_key).stem,
                project_root,
                encoded_dir,
                tile_param=tile_param,
                split_params=split_params,
            )

            # Select fragment.
            # Prefer pre-computed positions from the pipeline (scanned all images).
            _frag_key = str(resolution) if resolution is not None else "null"
            if (
                worst_fragments
                and "anisotropic" in worst_fragments
                and _frag_key in worst_fragments["anisotropic"].get("regions", {})
            ):
                frag = worst_fragments["anisotropic"]["regions"][_frag_key]
                region = WorstRegion(
                    x=frag["x"],
                    y=frag["y"],
                    width=frag["width"],
                    height=frag["height"],
                    avg_distortion=frag["avg_distortion"],
                )
                print(
                    f"  {f'[{res_label}] ' if res_label else ''}"
                    f"Using pre-computed fragment from pipeline"
                )
            else:
                region = find_worst_region_in_array(aniso_map, crop_size=config.crop_size)

            print(
                f"  {f'[{res_label}] ' if res_label else ''}"
                f"Region: ({region.x}, {region.y}) "
                f"{region.width}x{region.height} score={region.avg_distortion:.4f}"
            )

            # Track first resolution's data for the summary result
            if first_region is None:
                first_region = region
                first_worst_m = find_worst_measurement(source_measurements, metric=config.metric)
                if resolution_varies:
                    all_img_ms = [
                        m
                        for m in measurements
                        if m["original_image"] == img_key
                        and m.get("measurement_error") is None
                        and m.get(config.metric) is not None
                    ]
                    first_image_score = _anisotropic_image_score(
                        all_img_ms, config.metric, split_params
                    )
                else:
                    first_image_score = get_worst_image_score(
                        measurements,
                        img_key,
                        metric=config.metric,
                        split_params=split_params,
                    )

            # Build precomputed distmap lookup (keyed by id of measurement dict)
            precomputed_distmaps: dict[int, np.ndarray] = {
                id(m_dict): arr for arr, m_dict in variant_pairs
            }

            crops_dir = work / "crops" / (res_label or "original")
            crops_dir.mkdir(parents=True)

            # Crop original reference
            original_crop_path = crops_dir / "original.png"
            crop_and_zoom(
                source_path,
                region,
                zoom_factor=config.zoom_factor,
                output_path=original_crop_path,
            )

            crop_entries: list[tuple[Path, str, str]] = [
                (original_crop_path, "Original", "Reference"),
            ]

            # Sort: tile_param varies within figures; split_params across figures.
            _tile_key = tile_param or "format"
            sorted_measurements = sorted(
                source_measurements,
                key=lambda m: (m.get(_tile_key) or 0,) + tuple(m.get(p) or 0 for p in split_params),
            )

            comp_groups = _group_measurements_for_comparison(
                sorted_measurements, _tile_key, split_params
            )

            variant_distmap_entries: list[tuple[np.ndarray, str, str]] = []

            print(
                f"  {f'[{res_label}] ' if res_label else ''}"
                f"Obtaining {len(sorted_measurements)} variants "
                f"→ {len(comp_groups)} comparison figure(s)..."
            )
            for m in sorted_measurements:
                enc_path = _get_or_encode(original_path, m, encoded_dir, project_root)
                if enc_path is None:
                    continue

                label = _build_label(m, intra_res_varying)
                metric_label = _build_metric_label(m)
                safe_name = label.replace(" ", "_").replace("=", "-")
                crop_path = crops_dir / f"{safe_name}.png"

                crop_and_zoom(
                    enc_path,
                    region,
                    zoom_factor=config.zoom_factor,
                    output_path=crop_path,
                )
                crop_entries.append((crop_path, label, metric_label))

                if id(m) in precomputed_distmaps:
                    variant_distmap_entries.append(
                        (precomputed_distmaps[id(m)], label, metric_label)
                    )
                else:
                    variant_pfm = work / f"dm_{res_label or 'orig'}_{safe_name}.pfm"
                    try:
                        generate_distortion_map(source_path, enc_path, variant_pfm)
                        dm_arr = read_pfm(variant_pfm)
                        variant_distmap_entries.append((dm_arr, label, metric_label))
                    except (RuntimeError, ValueError) as exc:
                        print(f"    Warning: distortion map for {label} failed: {exc}")

            print(
                f"  {f'[{res_label}] ' if res_label else ''}"
                f"Cropped {len(crop_entries)} images (including original)"
            )

            # Assemble comparison grids — one figure per split-param group.
            output_images: list[Path] = []
            res_dir.mkdir(parents=True, exist_ok=True)

            for group_label, group_indices in comp_groups:
                group_entries = [crop_entries[0]] + [
                    crop_entries[1 + i] for i in group_indices if 1 + i < len(crop_entries)
                ]
                if len(group_entries) < 2:
                    continue
                grid_filename = (
                    "comparison.webp" if len(comp_groups) == 1 else f"comparison_{group_label}.webp"
                )
                grid_path = res_dir / grid_filename
                assemble_comparison_grid(
                    group_entries,
                    grid_path,
                    max_columns=config.max_columns,
                    label_font_size=config.label_font_size,
                )
                output_images.append(grid_path)
                print(
                    f"  {f'[{res_label}] ' if res_label else ''}"
                    f"Generated comparison grid: {grid_path}"
                )

            # Distortion map comparison grid
            if variant_distmap_entries:
                distmap_thumbs_dir = work / "distmap_thumbs" / (res_label or "original")
                distmap_thumbs_dir.mkdir(parents=True)
                target_side = config.crop_size * config.zoom_factor

                orig_thumb = distmap_thumbs_dir / "original.png"
                with Image.open(source_path) as _src:
                    _src.convert("RGB").resize(
                        (target_side, target_side),
                        Image.Resampling.LANCZOS,
                    ).save(orig_thumb)

                distmap_crop_entries: list[tuple[Path, str, str]] = [
                    (orig_thumb, "Original", "Reference image"),
                ]
                for idx, (dm_arr, dm_label, dm_metric) in enumerate(variant_distmap_entries):
                    safe_dm = dm_label.replace(" ", "_").replace("=", "-")
                    thumb_path = distmap_thumbs_dir / f"dm_{idx:03d}_{safe_dm}.png"
                    _render_distmap_thumbnail(
                        dm_arr,
                        target_side,
                        thumb_path,
                        vmax=config.distmap_vmax,
                    )
                    distmap_crop_entries.append((thumb_path, dm_label, dm_metric))

                for group_label, group_indices in comp_groups:
                    group_dm_entries = [distmap_crop_entries[0]] + [
                        distmap_crop_entries[1 + i]
                        for i in group_indices
                        if 1 + i < len(distmap_crop_entries)
                    ]
                    if len(group_dm_entries) < 2:
                        continue
                    dm_filename = (
                        "distortion_map_comparison.webp"
                        if len(comp_groups) == 1
                        else f"distortion_map_comparison_{group_label}.webp"
                    )
                    dm_grid_path = res_dir / dm_filename
                    assemble_comparison_grid(
                        group_dm_entries,
                        dm_grid_path,
                        max_columns=config.max_columns,
                        label_font_size=config.label_font_size,
                    )
                    output_images.append(dm_grid_path)
                    print(
                        f"  {f'[{res_label}] ' if res_label else ''}"
                        f"Generated distortion map grid: {dm_grid_path}"
                    )

            # Visualize the anisotropic distortion map with region annotation
            _visualize_distortion_map(
                aniso_map,
                region,
                res_dir / "distortion_map_anisotropic.webp",
                dash_color="cyan",
            )
            _save_annotated_original(
                source_path,
                region,
                res_dir / "original_annotated.webp",
            )

            all_output_images.extend(output_images)

        # Build result
        worst_m = first_worst_m or find_worst_measurement(
            [
                m
                for m in measurements
                if (m["original_image"] if resolution_varies else m["source_image"]) == img_key
                and m.get("measurement_error") is None
                and m.get(config.metric) is not None
            ],
            metric=config.metric,
        )

        strategy_results.append(
            StrategyResult(
                strategy="anisotropic",
                source_image=img_key,
                image_score=first_image_score if first_image_score is not None else 0.0,
                worst_format=worst_m["format"],
                worst_quality=worst_m["quality"],
                worst_metric_value=worst_m[config.metric],
                region=first_region or WorstRegion(0, 0, config.crop_size, config.crop_size, 0.0),
                output_dir=output_dir,
                output_images=all_output_images,
            )
        )

    print("\nComparison complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  {len(all_output_images)} comparison images generated")

    return ComparisonResult(
        study_id=study_id,
        strategies=strategy_results,
        varying_parameters=varying,
    )


def _render_distmap_thumbnail(
    arr: np.ndarray,
    target_size: int,
    output_path: Path,
    vmin: float = 0.0,
    vmax: float = 10.0,
) -> Path:
    """Render a full distortion map as a fixed-scale viridis thumbnail.

    The entire ``arr`` (full-image distortion values) is normalised using
    the fixed ``[vmin, vmax]`` range, mapped through the viridis colormap,
    and then scaled to ``target_size \u00d7 target_size`` pixels using
    high-quality (Lanczos) down-sampling.

    Using the same ``vmin``/``vmax`` for every tile in the comparison
    grid guarantees that equal colours represent equal levels of
    distortion across all encoding variants, so structural differences
    (e.g. one encoder concentrating errors in fine-detail regions while
    another spreads them evenly) are directly visible.

    Args:
        arr: Full-image 2-D float distortion array ``(H, W)``.
        target_size: Each output tile will be scaled to this width and
            height in pixels.  Pass ``config.crop_size * config.zoom_factor``
            so tiles match the dimensions of the pixel-crop comparison tiles.
        output_path: Destination PNG path.
        vmin: Distortion value mapped to the darkest viridis colour
            (default ``0.0``).
        vmax: Distortion value mapped to the brightest viridis colour.
            Values above this are clamped (default ``10.0``).

    Returns:
        ``output_path``.
    """
    if vmax > vmin:
        normalised = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        normalised = np.zeros_like(arr, dtype=np.float64)

    rgba = matplotlib.colormaps["viridis_r"](normalised)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb, mode="RGB")
    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_path)
    return output_path


def _draw_annotation_on_image(
    image_path: Path,
    region: WorstRegion,
    output_path: Path,
    *,
    dash_color: str = "black",
    label: str | None = None,
) -> None:
    """Draw a high-contrast dashed rectangle on an image using ImageMagick.

    Uses a thick white solid outer border followed by a thin dashed coloured
    inner border so the annotation remains visible against any background.
    When *label* is supplied a short text badge is drawn at the top-left
    corner of the rectangle on a white semi-transparent background.

    Args:
        image_path: Source image file (any format ImageMagick can read).
        region: The region to annotate.
        output_path: Destination path (can equal ``image_path`` for in-place).
        dash_color: Colour of the inner dashed border and label text.
        label: Optional short text rendered inside a badge at the top-left
            corner of the rectangle.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x1, y1 = region.x, region.y
    x2, y2 = region.x + region.width, region.y + region.height

    draw_cmd = (
        f"fill none "
        f"stroke white stroke-width 5 "
        f"rectangle {x1},{y1} {x2},{y2} "
        f"stroke-dasharray 10,6 "
        f"stroke {dash_color} stroke-width 2 "
        f"rectangle {x1},{y1} {x2},{y2}"
    )
    if label:
        tx, ty = x1 + 2, y1 + 2
        draw_cmd += (
            f" fill rgba(255,255,255,0.82) stroke none "
            f"roundrectangle {tx},{ty} {tx + 34},{ty + 18} 3,3 "
            f"fill {dash_color} font DejaVu-Sans font-size 13 stroke none "
            f"text {tx + 4},{ty + 14} '{label}'"
        )
    extra_args: list[str] = []
    if output_path.suffix.lower() == ".webp":
        extra_args = ["-define", "webp:lossless=true"]
    cmd = ["magick", str(image_path), "-draw", draw_cmd, *extra_args, str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: Could not annotate image: {result.stderr}")


def _visualize_distortion_map(
    arr: np.ndarray,
    region: WorstRegion,
    output_path: Path,
    *,
    dash_color: str = "black",
) -> None:
    """Render a distortion array as a viridis-coloured image with region annotation.

    The *viridis_r* colormap is perceptually uniform, colorblind-safe, and
    grayscale-compatible: bright-yellow pixels have low distortion/variance
    and dark-purple pixels have high distortion/variance.  The selected
    fragment is annotated via :func:`_draw_annotation_on_image`.

    Args:
        arr: 2-D float array of distortion values (any scale; will be
            normalised to ``[0, 1]`` before colouring).
        region: The region to annotate.
        output_path: Destination path (PNG or lossless WebP based on suffix).
        dash_color: Colour of the inner dashed border annotation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max > arr_min:
        normalised = (arr - arr_min) / (arr_max - arr_min)
    else:
        normalised = np.zeros_like(arr, dtype=np.float64)

    rgba = matplotlib.colormaps["viridis_r"](normalised)  # (H, W, 4) in [0, 1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb, mode="RGB")
    if output_path.suffix.lower() == ".webp":
        pil_img.save(output_path, format="WEBP", lossless=True)
    else:
        pil_img.save(output_path)

    _draw_annotation_on_image(output_path, region, output_path, dash_color=dash_color)


def _save_annotated_original(
    source_path: Path,
    region: WorstRegion,
    output_path: Path,
) -> None:
    """Save a copy of the source image with the selected region annotated.

    Args:
        source_path: Path to the original source image.
        region: The region to annotate.
        output_path: Destination image path.
    """
    _draw_annotation_on_image(source_path, region, output_path)
