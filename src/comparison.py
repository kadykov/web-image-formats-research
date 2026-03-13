"""Visual comparison module for identifying and visualizing encoding artifacts.

This module generates side-by-side comparison figures showing how different
encoding variants render the same region at matched quality or file-size
levels using interpolation-based quality matching.

The workflow is:

1. Load pipeline quality measurements from ``quality.json`` and
   comparison configuration from the study configuration file.
2. For each target group (e.g., ``ssimulacra2=[60,70,80]``), select the
   source image with highest cross-format coefficient of variation (CV =
   std / mean) of the output metric via interpolation.
3. Encode images at interpolated quality settings for **every** target
   value in the group, compute Butteraugli distortion maps, then
   compute a single aggregate anisotropic standard-deviation map across
   all target values.  This yields **one** fragment region per group.
4. Using the shared fragment, crop every variant at every target value
   and assemble labeled comparison grids with supplementary figures
   (distortion-map grids, annotated originals).

The comparison script reads its configuration (targets, tile parameter,
excluded images) directly from the study configuration JSON, not from
quality.json.  This allows tuning comparison parameters without
re-running the main pipeline.

This decouples the comparison figure generation from the pipeline:
the pipeline is a pure encode-and-measure step, while this module
independently selects images and quality settings via interpolation.

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
from src.interpolation import (
    _extract_fixed_params,
    interpolate_quality_for_metric,
    select_best_image,
)
from src.quality import (
    QualityMeasurer,
    WorstRegion,
    extract_fragment,
    find_worst_region_in_array,
    read_pfm,
    to_png,
)


@dataclass
class ComparisonConfig:
    """Configuration for visual comparison generation.

    Attributes:
        crop_size: Size of the crop region in original pixels (before zoom).
        zoom_factor: Factor to scale the crop (e.g., 3 for 300% zoom).
        max_columns: Maximum number of images per row in the grid.
        label_font_size: Font size for labels in the comparison grid.
        distmap_vmax: Upper bound of the fixed Butteraugli distortion
            scale used in the distortion-map comparison grid.  All
            per-pixel values are clamped to ``[0, distmap_vmax]`` before
            mapping to the viridis colormap, ensuring every tile uses an
            identical colour scale so structural differences between
            encoding variants are directly comparable.
            Defaults to ``5.0``.
        source_image: Optional explicit source image path (relative to
            project root) to use instead of automatic selection.
        tile_parameter: The encoding parameter that should vary within
            each comparison figure — i.e. each tile shows a different
            value of this parameter.

            When ``None`` the value is taken from the study
            configuration file.  If that is also absent the built-in
            heuristic is used: ``"format"`` when multiple formats vary,
            otherwise the first non-quality sweep parameter.
        study_config_path: Path to the study configuration JSON file.
            When provided, comparison targets, tile parameter, and
            excluded images are read from this file instead of from
            quality.json metadata.
    """

    crop_size: int = 128
    zoom_factor: int = 3
    max_columns: int = 4
    label_font_size: int = 22
    distmap_vmax: float = 5.0
    source_image: str | None = None
    tile_parameter: str | None = None
    study_config_path: Path | None = None


@dataclass
class TargetComparisonResult:
    """Result for one target-value comparison figure.

    Attributes:
        target_metric: The metric being matched (e.g. ``"ssimulacra2"``).
        target_value: The target value (e.g. ``70``).
        source_image: Path (relative to project root) of the selected
            source image.
        region: The detected worst fragment coordinates.
        interpolated_qualities: Mapping from format name to the
            interpolated encoder quality setting used.
        output_images: List of generated comparison image paths.
    """

    target_metric: str
    target_value: float
    source_image: str
    region: WorstRegion
    interpolated_qualities: dict[str, float]
    output_images: list[Path] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of the visual comparison generation.

    Attributes:
        study_id: Study identifier.
        target_results: List of per-target-value results.
        varying_parameters: Parameters that vary across measurements.
    """

    study_id: str
    target_results: list[TargetComparisonResult] = field(default_factory=list)
    varying_parameters: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reusable utility functions
# ---------------------------------------------------------------------------


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


def sort_tile_values(raw: set[str] | list[str]) -> list[str]:
    """Sort tile-parameter value strings numerically when possible.

    Numeric values (e.g. effort levels ``["1", "2", ..., "10"]``) are
    sorted as floats so that ``"10"`` comes after ``"9"`` rather than
    after ``"1"`` (lexicographic order).  Non-numeric strings fall back
    to plain lexicographic sort.

    Args:
        raw: Collection of tile-parameter value strings.

    Returns:
        Sorted list of value strings.
    """
    try:
        return sorted(raw, key=lambda v: (float(v), v))
    except ValueError:
        return sorted(raw)


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
        "crop",
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
        Label string like "SSIM2:75.5 BA:2.5 BPP:0.80"
    """
    parts = []
    if measurement.get("ssimulacra2") is not None:
        parts.append(f"SSIM2:{measurement['ssimulacra2']:.1f}")
    if measurement.get("butteraugli") is not None:
        parts.append(f"BA:{measurement['butteraugli']:.2f}")
    bits_per_pixel = measurement.get("bits_per_pixel")
    if bits_per_pixel is None:
        file_size = measurement.get("file_size")
        width = measurement.get("width")
        height = measurement.get("height")
        if file_size is not None and width and height:
            bits_per_pixel = 8 * file_size / (width * height)
    if bits_per_pixel is not None:
        parts.append(f"BPP:{bits_per_pixel:.3g}")
    return " ".join(parts)


def _format_figure_title(metric: str, value: float) -> str:
    """Format a human-readable figure title from a metric name and target value.

    Args:
        metric: Metric identifier (e.g. ``"ssimulacra2"``, ``"bits_per_pixel"``).
        value: Target value for the metric.

    Returns:
        Title string like ``"Target: SSIMULACRA2 = 75"`` or
        ``"Target: BPP = 0.10"``.
    """
    _NAMES = {
        "ssimulacra2": "SSIMULACRA2",
        "bits_per_pixel": "BPP",
        "psnr": "PSNR",
        "ssim": "SSIM",
        "butteraugli": "Butteraugli",
    }
    name = _NAMES.get(metric, metric.replace("_", " ").title())
    val_str = f"{value:g}"
    return f"Target: {name} = {val_str}"


def assemble_comparison_grid(
    crops: list[tuple[Path, str, str]],
    output_path: Path,
    max_columns: int = 4,
    label_font_size: int = 22,
    figure_title: str | None = None,
    placeholder_indices: frozenset[int] | None = None,
) -> Path:
    """Assemble cropped images into a labeled grid using ImageMagick montage.

    Each image is annotated with a label (encoding parameters and metrics)
    placed **below** the tile.  When *figure_title* is given, a centred
    title row is prepended above the grid.  Tiles listed in
    *placeholder_indices* are rendered with white-on-white invisible labels
    so they appear as blank spacers, preserving the grid layout when some
    variants are unavailable at a given quality target.

    Args:
        crops: List of (image_path, title_label, metric_label) tuples.
        output_path: Path where the grid image will be saved.
        max_columns: Maximum images per row.
        label_font_size: Font size for per-tile labels.
        figure_title: Optional title rendered above the whole grid.
        placeholder_indices: 0-based indices into *crops* whose tiles
            should be rendered as invisible spacers (white image, white
            text labels).

    Returns:
        Path to the generated comparison grid image.

    Raises:
        RuntimeError: If montage command fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine grid geometry
    n = len(crops)
    cols = min(n, max_columns)

    _placeholder_set = placeholder_indices or frozenset()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create labeled tiles using ImageMagick – labels placed below each tile
        labeled_paths: list[str] = []
        for i, (img_path, title, metrics) in enumerate(crops):
            labeled_path = tmpdir_path / f"labeled_{i:03d}.png"
            is_placeholder = i in _placeholder_set
            text_color = "white" if is_placeholder else "black"

            # Place labels at the bottom (South) of each tile
            combined_label = f"{title}\\n{metrics}"
            cmd = [
                "magick",
                str(img_path),
                "-gravity",
                "South",
                "-background",
                "white",
                "-splice",
                f"0x{label_font_size * 3}",
                "-font",
                "DejaVu-Sans",
                "-pointsize",
                str(label_font_size),
                "-fill",
                text_color,
                "-gravity",
                "South",
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

    # Prepend a figure title row above the assembled grid
    if figure_title is not None:
        title_font_size = label_font_size + 6
        title_height = title_font_size * 2
        lossless_title = (
            ["-define", "webp:lossless=true"] if output_path.suffix.lower() == ".webp" else []
        )
        title_cmd = [
            "magick",
            str(output_path),
            "-gravity",
            "North",
            "-background",
            "white",
            "-splice",
            f"0x{title_height}",
            "-font",
            "DejaVu-Sans",
            "-pointsize",
            str(title_font_size),
            "-fill",
            "black",
            "-gravity",
            "North",
            "-annotate",
            f"+0+{(title_height - title_font_size) // 2}",
            figure_title,
            *lossless_title,
            str(output_path),
        ]
        title_result = subprocess.run(title_cmd, capture_output=True, text=True)
        if title_result.returncode != 0:
            print(f"  Warning: Could not add figure title: {title_result.stderr}")

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

    When the measurement includes a ``crop`` parameter and
    ``analysis_fragment`` / ``crop_region``, the source image is cropped
    accordingly before encoding.

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
    crop_level = measurement.get("crop")
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
    elif crop_level is not None:
        crop_region = measurement.get("crop_region")
        if crop_region is not None:
            import tempfile as _tempfile

            _tmp_dir = _tmp_dir or _tempfile.mkdtemp(prefix="comparison_crop_")
            # Crop using stored crop_region coordinates
            from PIL import Image as _PILImg

            cx, cy = crop_region["x"], crop_region["y"]
            cw, ch = crop_region["width"], crop_region["height"]
            with _PILImg.open(source_path) as _im:
                cropped = _im.crop((cx, cy, cx + cw, cy + ch))
                crop_path = Path(_tmp_dir) / f"{source_path.stem}_c{crop_level}.png"
                cropped.save(crop_path)
                actual_source = crop_path

    encoder = ImageEncoder(output_dir)
    fmt = measurement["format"]
    quality = measurement["quality"]
    output_name = f"encoded_{fmt}_q{quality}"

    # Add extra params to output name for uniqueness
    for param in ("chroma_subsampling", "speed", "effort", "method", "resolution", "crop"):
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
        cache: Mutable dict mapping resolution to preprocessed path.
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


def _resolve_source_for_crop(
    original_path: Path,
    crop_level: int | None,
    measurements: list[dict],
    cache: dict[int, tuple[Path, dict, dict | None]],
    tmpdir: Path,
    *,
    selected_image: str | None = None,
) -> tuple[Path, dict | None, dict | None]:
    """Get the correct source image for a given crop level.

    For ``crop_level=None`` returns the original image with no fragment/region.
    For a specific crop level, crops the original using stored ``crop_region``
    coordinates from the measurement data, caching the result.

    Args:
        original_path: Path to the original dataset image.
        crop_level: Target longest-edge for the crop, or ``None``.
        measurements: Measurement dicts from the quality results.
        cache: Mutable dict mapping crop level to (path, crop_region, analysis_fragment).
        tmpdir: Temporary directory for cropped copies.
        selected_image: When provided, only measurements whose
            ``original_image`` (or ``source_image``) matches this value
            are considered.  This prevents using crop regions computed
            for a different image (which would have wrong coordinates
            and a different aspect ratio).

    Returns:
        Tuple of (source_path, crop_region_dict, analysis_fragment_dict).
    """
    if crop_level is None:
        return original_path, None, None
    if crop_level in cache:
        return cache[crop_level]

    # Find a measurement with this crop level that has crop_region info,
    # restricted to the selected image so we don't pick up coordinates
    # from a different image with a different aspect ratio.
    def _matches_image(m: dict) -> bool:
        if selected_image is None:
            return True
        img_path: str = m.get("original_image", m.get("source_image", ""))
        return img_path == selected_image

    example = next(
        (
            m
            for m in measurements
            if m.get("crop") == crop_level
            and m.get("crop_region") is not None
            and _matches_image(m)
        ),
        None,
    )
    if example is None:
        # No crop_region stored — fall back to using preprocessing
        from src.preprocessing import ImagePreprocessor

        crop_dir = tmpdir / f"c{crop_level}"
        preprocessor = ImagePreprocessor(crop_dir)
        # Use a dummy fragment center (image center) when no data available
        from PIL import Image as _PILImg

        with _PILImg.open(original_path) as _im:
            frag_size = 200
            frag_x = max(0, _im.width // 2 - frag_size // 2)
            frag_y = max(0, _im.height // 2 - frag_size // 2)
            analysis_fragment = {
                "x": frag_x,
                "y": frag_y,
                "width": min(frag_size, _im.width),
                "height": min(frag_size, _im.height),
            }
        result = preprocessor.crop_image_around_fragment(
            original_path,
            fragment=analysis_fragment,
            target_longest_edge=crop_level,
        )
        cache[crop_level] = (result.path, result.crop_region, analysis_fragment)
        return cache[crop_level]

    crop_region: dict[str, int] = example["crop_region"]
    source_analysis_fragment: dict[str, int] | None = example.get("analysis_fragment")

    from PIL import Image as _PILImg

    cx, cy = crop_region["x"], crop_region["y"]
    cw, ch = crop_region["width"], crop_region["height"]

    crop_dir = tmpdir / f"c{crop_level}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{original_path.stem}_c{crop_level}.png"

    if not crop_path.exists():
        with _PILImg.open(original_path) as _im:
            cropped = _im.crop((cx, cy, cx + cw, cy + ch))
            cropped.save(crop_path)

    cache[crop_level] = (crop_path, crop_region, source_analysis_fragment)
    return cache[crop_level]


def _analysis_fragment_in_original(
    crop_cache: dict[int, tuple[Path, dict, dict | None]],
    measurements: list[dict],
    *,
    selected_image: str | None = None,
) -> dict[str, int] | None:
    """Return the analysis fragment rectangle in original image coordinates.

    Checks the crop_cache first (populated by ``_resolve_source_for_crop``),
    then falls back to scanning measurements.

    Args:
        crop_cache: Crop level → (path, crop_region, analysis_fragment).
        measurements: Measurement dicts from the quality results.
        selected_image: When provided, only measurements whose
            ``original_image`` (or ``source_image``) matches this value
            are considered in the fallback scan.

    Returns:
        ``{"x", "y", "width", "height"}`` in original-image coordinates,
        or ``None`` if no fragment information is available.
    """
    # Try crop_cache entries (crop_region is in original coords,
    # analysis_fragment is in crop coords).
    for _crop_level, (_path, cr, af) in crop_cache.items():
        if cr is not None and af is not None:
            cr_dict: dict[str, int] = cr
            af_dict: dict[str, int] = af
            return {
                "x": cr_dict["x"] + af_dict["x"],
                "y": cr_dict["y"] + af_dict["y"],
                "width": af_dict["width"],
                "height": af_dict["height"],
            }

    # Fallback: scan measurements for crop_region + analysis_fragment,
    # restricted to the selected image.
    for m in measurements:
        if selected_image is not None:
            img = m.get("original_image", m.get("source_image", ""))
            if img != selected_image:
                continue
        crop_region_m: dict[str, int] | None = m.get("crop_region")
        analysis_fragment_m: dict[str, int] | None = m.get("analysis_fragment")
        if crop_region_m is not None and analysis_fragment_m is not None:
            return {
                "x": crop_region_m["x"] + analysis_fragment_m["x"],
                "y": crop_region_m["y"] + analysis_fragment_m["y"],
                "width": analysis_fragment_m["width"],
                "height": analysis_fragment_m["height"],
            }

    return None


def _anisotropic_std_map(
    variant_pairs: list[tuple[np.ndarray, dict]],
    split_params: list[str],
) -> np.ndarray:
    """Compute the anisotropic standard-deviation distortion map for fragment selection.

    Groups variant distortion arrays by *split_params* (one group per
    comparison figure), computes pixel-wise standard deviation within
    each group across the tile-parameter values, then averages these
    per-group std maps.

    Standard deviation (rather than CV) is used here because we
    *want* to favour visually prominent, high-distortion regions:
    these are the most noticeable to human observers and therefore
    the most informative fragments to show.  CV would normalise away
    the absolute magnitude and risk selecting dark, featureless areas.

    Falls back to overall pixel-wise std when no split group contains
    >=2 variants (i.e. the study sweeps only a single parameter and
    all variants go on one figure).

    Args:
        variant_pairs: List of ``(distortion_array, measurement_dict)``
            for every encoding variant that was successfully measured.
        split_params: Parameters whose distinct value combinations
            define the groups.

    Returns:
        2-D ``float64`` array of shape ``(H, W)`` with anisotropic
        standard-deviation values.  Higher values indicate regions
        where the tile-parameter choice produces the most spread in
        absolute distortion.
    """

    def _std(stacked: np.ndarray) -> np.ndarray:
        """Pixel-wise standard deviation."""
        return stacked.std(axis=0)  # type: ignore[no-any-return]

    # Group arrays by split_params key
    groups: dict[tuple, list[np.ndarray]] = {}
    for arr, m in variant_pairs:
        key = tuple(str(m.get(p)) for p in split_params)
        groups.setdefault(key, []).append(arr)

    # Compute within-group pixel-wise std for groups with >=2 variants
    group_std_maps: list[np.ndarray] = []
    for arrs in groups.values():
        if len(arrs) >= 2:
            stacked = np.stack(arrs, axis=0)  # (K, H, W)
            group_std_maps.append(_std(stacked))

    if group_std_maps:
        return np.stack(group_std_maps, axis=0).mean(axis=0)  # type: ignore[no-any-return]

    # Fallback: overall std (when single-tile_param study)
    all_arrs = [arr for arr, _ in variant_pairs]
    if len(all_arrs) >= 2:
        stacked = np.stack(all_arrs, axis=0)
        return _std(stacked)

    # Single variant: return the distortion map itself
    return variant_pairs[0][0].astype(np.float64)


def _default_tile_parameter(varying: list[str]) -> str | None:
    """Determine the default tile parameter from the set of varying parameters.

    Heuristic (in priority order):

    1. If ``"format"`` varies -> ``"format"`` (cross-encoder studies).
    2. If any non-quality parameter varies -> the first such parameter.
    3. Otherwise -> ``"quality"``.

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_comparison(
    quality_json_path: Path,
    output_dir: Path,
    project_root: Path,
    config: ComparisonConfig | None = None,
) -> ComparisonResult:
    """Generate visual comparison images using interpolation-based matching.

    For each target group defined in the study configuration (e.g.
    ``ssimulacra2=[60,70,80]`` or ``bits_per_pixel=[0.8,2.4,4.0]``),
    this function:

    1. Selects the source image with highest cross-format coefficient of
       variation (CV = std / mean) of the output metric
       (via :func:`src.interpolation.select_best_image`), respecting any
       image exclusions from the study config.
    2. For **all** target values in the group, interpolates encoder
       quality per format, encodes, and computes distortion maps.
    3. Computes a **single** aggregate anisotropic standard-deviation map
       across all target values, yielding one fragment region per group.
    4. Using the shared fragment, crops every variant at every target
       value and assembles labeled comparison grids and supplementary
       figures (one distortion-map + annotated original per group).

    The comparison configuration (targets, tile parameter, excluded
    images) is read from the study configuration file when
    ``config.study_config_path`` is set.

    Args:
        quality_json_path: Path to the quality.json results file.
        output_dir: Directory where comparison images will be saved.
        project_root: Project root directory for resolving relative paths.
        config: Comparison configuration (uses defaults if ``None``).

    Returns:
        :class:`ComparisonResult` with per-target results.

    Raises:
        FileNotFoundError: If quality results or source images are not found.
        RuntimeError: If image processing tools fail.
    """
    if config is None:
        config = ComparisonConfig()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load quality results
    data = load_quality_results(quality_json_path)
    measurements = data["measurements"]
    study_id = data.get("study_id", "unknown")

    print(f"Generating visual comparison for study: {study_id}")
    print(f"  Total measurements: {len(measurements)}")

    # 2. Load study config for comparison parameters
    from src.study import StudyConfig

    study_config: StudyConfig | None = None
    if config.study_config_path is not None:
        study_config = StudyConfig.from_file(config.study_config_path)

    # 3. Resolve targets
    targets: list[dict] = []
    if study_config is not None and study_config.comparison_targets:
        targets = study_config.comparison_targets
    if not targets:
        targets = [{"metric": "ssimulacra2", "values": [60, 70, 80]}]
        print("  No comparison targets configured; using defaults")

    # 4. Resolve exclude images
    exclude_images: list[str] | None = None
    if study_config is not None:
        exclude_images = study_config.comparison_exclude_images

    # 5. Determine varying parameters and tile parameter
    varying = determine_varying_parameters(measurements)

    tile_param: str | None = config.tile_parameter
    if tile_param is None and study_config is not None:
        tile_param = study_config.comparison_tile_parameter
    if tile_param is None:
        tile_param = _default_tile_parameter(varying)

    print(f"  Tile parameter: {tile_param!r}")
    print(f"  Targets: {len(targets)} group(s)")
    for tg in targets:
        print(f"    {tg['metric']}: {tg['values']}")
    if exclude_images:
        print(f"  Excluded images: {exclude_images}")

    # 6. Collect unique tile-parameter values, sorted numerically when possible
    _tile_values_raw = {
        str(m.get(tile_param)) for m in measurements if m.get(tile_param) is not None
    }
    tile_values = sort_tile_values(_tile_values_raw)

    # Detect resolution variation
    resolution_varies = "resolution" in varying
    unique_resolutions: list[int | None]
    if resolution_varies:
        unique_resolutions = sorted(
            {m.get("resolution") for m in measurements if m.get("resolution") is not None}
        )
    else:
        unique_resolutions = [None]

    # Detect crop variation
    crop_varies = "crop" in varying
    unique_crops: list[int | None]
    if crop_varies and tile_param != "crop":
        unique_crops = sorted({m.get("crop") for m in measurements if m.get("crop") is not None})
    else:
        unique_crops = [None]

    intra_res_varying = (
        [p for p in varying if p not in ("resolution", "crop")]
        if (resolution_varies or crop_varies)
        else varying
    )

    target_results: list[TargetComparisonResult] = []

    with tempfile.TemporaryDirectory() as _work:
        work = Path(_work)

        for target_group in targets:
            # Reset per-image caches at every target group because each
            # group may select a different source image.
            preprocess_cache: dict[int, Path] = {}
            crop_cache: dict[int, tuple[Path, dict, dict | None]] = {}
            _af_frag: dict[str, int] | None = None
            target_metric: str = target_group["metric"]
            target_values: list[float] = target_group["values"]

            # Output metric for image selection (the "other" metric)
            if target_metric in ("ssimulacra2", "psnr", "ssim", "butteraugli"):
                output_metric = "bits_per_pixel"
            else:
                output_metric = "ssimulacra2"

            # Select source image (once per target group)
            selected_image: str | None
            if config.source_image is not None:
                selected_image = config.source_image
                print(f"  [{target_metric}] Using explicit source image: {selected_image}")
            else:
                selected_image = select_best_image(
                    measurements,
                    tile_param or "format",
                    target_metric,
                    target_values,
                    output_metric,
                    exclude_images=exclude_images,
                )
                if selected_image is None:
                    # Fallback: pick the image with the most valid measurements
                    valid = [m for m in measurements if m.get("measurement_error") is None]
                    if exclude_images:
                        excluded_set = set(exclude_images)
                        valid = [
                            m
                            for m in valid
                            if Path(m.get("original_image", m.get("source_image", ""))).name
                            not in excluded_set
                        ]
                    if valid:
                        from collections import Counter

                        img_counts = Counter(
                            m.get("original_image", m.get("source_image", "")) for m in valid
                        )
                        selected_image = img_counts.most_common(1)[0][0]
                        print(
                            f"  [{target_metric}] Interpolation-based selection "
                            f"unavailable; falling back to: {selected_image}"
                        )
                    else:
                        print(f"  [{target_metric}] No valid measurements, skipping")
                        continue
                else:
                    print(f"  [{target_metric}] Selected image: {selected_image}")
            assert selected_image is not None

            original_path = project_root / selected_image
            if not original_path.exists():
                msg = f"Source image not found: {original_path}"
                raise FileNotFoundError(msg)

            for resolution in unique_resolutions:
                res_source_path = _resolve_source_for_resolution(
                    original_path,
                    resolution,
                    preprocess_cache,
                    work / "prep",
                )

                for crop_level in unique_crops:
                    crop_source_path: Path
                    crop_region_info: dict | None = None
                    analysis_fragment_info: dict | None = None

                    if crop_level is not None:
                        (
                            crop_source_path,
                            crop_region_info,
                            analysis_fragment_info,
                        ) = _resolve_source_for_crop(
                            original_path,
                            crop_level,
                            measurements,
                            crop_cache,
                            work / "prep",
                            selected_image=selected_image,
                        )
                    else:
                        crop_source_path = res_source_path

                    source_path = crop_source_path

                    # Filter measurements to the current resolution/crop level
                    level_measurements = measurements
                    if resolution is not None:
                        level_measurements = [
                            m for m in level_measurements if m.get("resolution") == resolution
                        ]
                    if crop_level is not None:
                        level_measurements = [
                            m for m in level_measurements if m.get("crop") == crop_level
                        ]

                    # Build group label
                    res_label = f"r{resolution}" if resolution is not None else None
                    crop_label = f"c{crop_level}" if crop_level is not None else None
                    group_label = target_metric
                    level_prefix = res_label or crop_label
                    if level_prefix:
                        group_label = f"{level_prefix}/{group_label}"

                    # ----------------------------------------------------------
                    # Phase 1: Encode all target values, collect distortion maps
                    # ----------------------------------------------------------
                    # per_value_data[target_value] = list of (enc_path, measurement, dm_arr)
                    # enc_path is None for placeholder entries (quality out of range).
                    per_value_data: dict[
                        float, list[tuple[Path | None, dict, np.ndarray | None]]
                    ] = {}
                    per_value_qualities: dict[float, dict[str, float]] = {}

                    for target_value in target_values:
                        target_label = f"{target_metric}_{target_value}"

                        print(f"  [{group_label}/{target_value}] Interpolating quality settings...")

                        encoded_variants: list[tuple[Path | None, dict, np.ndarray | None]] = []
                        interpolated_qualities: dict[str, float] = {}

                        for tv in tile_values:
                            example = next(
                                (
                                    m
                                    for m in level_measurements
                                    if str(m.get(tile_param)) == tv
                                    and m.get("measurement_error") is None
                                ),
                                None,
                            )
                            if example is None:
                                continue

                            fmt = example["format"]
                            fixed_kwargs = _extract_fixed_params(example, tile_param or "format")

                            if resolution is not None:
                                fixed_kwargs["resolution"] = resolution
                            if crop_level is not None and tile_param != "crop":
                                fixed_kwargs["crop"] = crop_level

                            # When the tile parameter is one of the encoder-level filter
                            # params (effort, speed, method, chroma_subsampling, resolution)
                            # it must be passed explicitly to interpolation so that the
                            # lookup is restricted to measurements for *this* specific tile
                            # value rather than being averaged across all tile values.
                            # 'format' is already passed as the `fmt` positional argument;
                            # 'quality' is the target of the interpolation and cannot be
                            # used as a filter — both are excluded here.
                            _ENCODER_FILTER_PARAMS = {
                                "speed",
                                "effort",
                                "method",
                                "chroma_subsampling",
                                "resolution",
                                "crop",
                            }
                            extra_tile_kwargs: dict = {}
                            if tile_param is not None and tile_param in _ENCODER_FILTER_PARAMS:
                                tile_val = example.get(tile_param)
                                if tile_val is not None:
                                    extra_tile_kwargs[tile_param] = tile_val

                            quality = interpolate_quality_for_metric(
                                level_measurements,
                                fmt,
                                target_metric,
                                target_value,
                                source_image=selected_image,
                                **fixed_kwargs,
                                **extra_tile_kwargs,
                            )
                            if quality is None:
                                print(
                                    f"    {fmt} ({tile_param}={tv}): target "
                                    f"{target_metric}={target_value} "
                                    f"out of measured range, adding placeholder"
                                )
                                placeholder_m: dict = {"format": fmt, "_placeholder": True}
                                if tile_param is not None:
                                    placeholder_m[tile_param] = example.get(tile_param) or tv
                                for _pp in ("speed", "effort", "method", "chroma_subsampling"):
                                    if example.get(_pp) is not None:
                                        placeholder_m[_pp] = example[_pp]
                                if resolution is not None:
                                    placeholder_m["resolution"] = resolution
                                if crop_level is not None:
                                    placeholder_m["crop"] = crop_level
                                encoded_variants.append((None, placeholder_m, None))
                                continue

                            # Key by tile value (tv) rather than format so that multiple
                            # tile values sharing the same format (e.g. different JXL
                            # effort levels) each get their own entry.
                            interpolated_qualities[tv] = quality
                            rounded_q = round(quality)

                            enc_measurement: dict = {
                                "format": fmt,
                                "quality": rounded_q,
                                "source_image": selected_image,
                            }
                            for param in ("speed", "effort", "method", "chroma_subsampling"):
                                if example.get(param) is not None:
                                    enc_measurement[param] = example[param]
                            if resolution is not None:
                                enc_measurement["resolution"] = resolution

                            # Resolve per-tile source for crop-tile studies.
                            # When crop is the tile parameter each tile encodes
                            # a different crop of the original image.
                            tile_ref_path = source_path
                            if tile_param == "crop":
                                tile_crop_val = int(tv)
                                enc_measurement["crop"] = tile_crop_val
                                tile_ref_path, tile_cr, _ = _resolve_source_for_crop(
                                    original_path,
                                    tile_crop_val,
                                    measurements,
                                    crop_cache,
                                    work / "prep",
                                    selected_image=selected_image,
                                )
                                # Do NOT set crop_region in enc_measurement:
                                # tile_ref_path is already cropped and
                                # encode_image would re-crop if it saw both.
                            elif crop_level is not None:
                                enc_measurement["crop"] = crop_level
                                # Do NOT set crop_region in enc_measurement:
                                # source_path (tile_ref_path) is already the
                                # cropped image; encode_image would re-crop if
                                # it saw crop_region (coordinates are in
                                # original-image space).

                            encoded_dir = work / "encoded" / target_label
                            encoded_dir.mkdir(parents=True, exist_ok=True)
                            enc_path = encode_image(tile_ref_path, enc_measurement, encoded_dir)
                            if enc_path is None:
                                continue

                            print(
                                f"    {fmt} ({tile_param}={tv}): "
                                f"q={rounded_q} (interpolated from {quality:.1f})"
                            )

                            enc_measurement["file_size"] = enc_path.stat().st_size
                            with Image.open(tile_ref_path) as _img:
                                enc_measurement["width"] = _img.width
                                enc_measurement["height"] = _img.height
                            enc_measurement["bits_per_pixel"] = (
                                8
                                * enc_measurement["file_size"]
                                / (enc_measurement["width"] * enc_measurement["height"])
                            )

                            # Measure quality metrics and generate the distortion map
                            # directly from the encoded file.  The encoded file already
                            # exists on disk, so using actual measurements is both more
                            # accurate and simpler than interpolating from the pipeline
                            # data — interpolation was only an approximation that also
                            # produced circular labels when the metric target matched the
                            # interpolated value exactly.
                            #
                            # For crop-tile studies the pipeline measures quality
                            # on the 200×200 analysis fragment only (not the full
                            # cropped image).  We replicate that here so that the
                            # distortion maps are also fragment-sized and can be
                            # meaningfully compared / aggregated across crop levels.
                            measure_ref_path = tile_ref_path
                            measure_enc_path = enc_path
                            if tile_param == "crop":
                                _tile_cache_entry = crop_cache.get(int(tv))
                                _af = (
                                    _tile_cache_entry[2] if _tile_cache_entry is not None else None
                                )
                                if _af is not None:
                                    _frag_dir = work / "fragments" / target_label / f"crop-{tv}"
                                    _frag_dir.mkdir(parents=True, exist_ok=True)
                                    _fx, _fy = _af["x"], _af["y"]
                                    _fw, _fh = _af["width"], _af["height"]

                                    _frag_ref = _frag_dir / f"ref_q{rounded_q}.png"
                                    with Image.open(tile_ref_path) as _rim:
                                        _rim.crop((_fx, _fy, _fx + _fw, _fy + _fh)).save(_frag_ref)
                                    measure_ref_path = _frag_ref

                                    _frag_enc = _frag_dir / f"enc_{fmt}_q{rounded_q}.png"
                                    extract_fragment(enc_path, _af, _frag_enc)
                                    measure_enc_path = _frag_enc

                            pfm_path = (
                                work
                                / "pfms"
                                / target_label
                                / f"distmap_{fmt}_{tile_param}-{tv}_q{rounded_q}.pfm"
                            )
                            pfm_path.parent.mkdir(parents=True, exist_ok=True)
                            measurer = QualityMeasurer()
                            dm_arr: np.ndarray | None = None
                            try:
                                metrics = measurer.measure_all(
                                    measure_ref_path, measure_enc_path, distmap_path=pfm_path
                                )
                                if metrics.ssimulacra2 is not None:
                                    enc_measurement["ssimulacra2"] = metrics.ssimulacra2
                                if metrics.butteraugli is not None:
                                    enc_measurement["butteraugli"] = metrics.butteraugli
                                if metrics.psnr is not None:
                                    enc_measurement["psnr"] = metrics.psnr
                                if metrics.ssim is not None:
                                    enc_measurement["ssim"] = metrics.ssim
                                if pfm_path.exists():
                                    dm_arr = read_pfm(pfm_path)
                            except (RuntimeError, ValueError, OSError) as exc:
                                print(
                                    f"    Warning: measurement for {fmt} ({tile_param}={tv}) failed: {exc}"
                                )

                            encoded_variants.append((enc_path, enc_measurement, dm_arr))

                        per_value_data[target_value] = encoded_variants
                        per_value_qualities[target_value] = interpolated_qualities

                    # ----------------------------------------------------------
                    # Phase 2: Compute single shared fragment for the group
                    # ----------------------------------------------------------

                    region: WorstRegion
                    aggregate_map: np.ndarray | None = None

                    if tile_param == "crop":
                        # Crop-tile studies: the comparison region is derived
                        # from the analysis fragment position in the original
                        # image so that crop-and-zoom works across different
                        # crop levels.
                        _af_frag = _analysis_fragment_in_original(
                            crop_cache,
                            measurements,
                            selected_image=selected_image,
                        )
                        if _af_frag is None:
                            print(
                                f"  [{group_label}] No analysis fragment info available, skipping"
                            )
                            continue
                        # Center the comparison crop within the analysis fragment
                        _coff_x = max(0, (_af_frag["width"] - config.crop_size) // 2)
                        _coff_y = max(0, (_af_frag["height"] - config.crop_size) // 2)
                        region = WorstRegion(
                            x=_af_frag["x"] + _coff_x,
                            y=_af_frag["y"] + _coff_y,
                            width=min(config.crop_size, _af_frag["width"]),
                            height=min(config.crop_size, _af_frag["height"]),
                            avg_distortion=0.0,
                        )

                        # Distortion maps are now fragment-sized (all 200×200)
                        # so they CAN be aggregated for the std-dev visualisation.
                        _crop_aniso_maps: list[np.ndarray] = []
                        for _tv_val, variants in per_value_data.items():
                            _vp = [(arr, m) for _, m, arr in variants if arr is not None]
                            if len(_vp) >= 2:  # noqa: PLR2004
                                _crop_aniso_maps.append(_anisotropic_std_map(_vp, []))
                        if not _crop_aniso_maps:
                            _all_p = [
                                (arr, m)
                                for variants in per_value_data.values()
                                for _, m, arr in variants
                                if arr is not None
                            ]
                            if len(_all_p) >= 2:  # noqa: PLR2004
                                _crop_aniso_maps = [_anisotropic_std_map(_all_p, [])]
                        if _crop_aniso_maps:
                            aggregate_map = np.stack(_crop_aniso_maps, axis=0).mean(axis=0)
                    else:
                        # Standard path: aggregate distortion maps for fragment
                        # selection.
                        per_value_aniso_maps: list[np.ndarray] = []
                        for _target_value, variants in per_value_data.items():
                            variant_pairs = [(arr, m) for _, m, arr in variants if arr is not None]
                            if len(variant_pairs) >= 2:  # noqa: PLR2004
                                aniso = _anisotropic_std_map(variant_pairs, [])
                                per_value_aniso_maps.append(aniso)

                        if not per_value_aniso_maps:
                            all_pairs = [
                                (arr, m)
                                for variants in per_value_data.values()
                                for _, m, arr in variants
                                if arr is not None
                            ]
                            if len(all_pairs) >= 2:  # noqa: PLR2004
                                per_value_aniso_maps = [_anisotropic_std_map(all_pairs, [])]

                        if not per_value_aniso_maps:
                            print(f"  [{group_label}] No distortion maps available, skipping")
                            continue

                        aggregate_map = np.stack(per_value_aniso_maps, axis=0).mean(axis=0)
                        if aggregate_map is None:
                            continue
                        region = find_worst_region_in_array(
                            aggregate_map, crop_size=config.crop_size
                        )

                    print(
                        f"  [{group_label}] Shared region: ({region.x}, {region.y}) "
                        f"{region.width}x{region.height} "
                        f"score={region.avg_distortion:.4f}"
                    )

                    # Determine the group output directory
                    if level_prefix:
                        group_dir = output_dir / level_prefix / target_metric
                    else:
                        group_dir = output_dir / target_metric
                    group_dir.mkdir(parents=True, exist_ok=True)

                    # Save supplementary figures (once per group)
                    if aggregate_map is not None:
                        # For crop-tile studies the aggregate map is in
                        # fragment-local coordinates (200×200), so translate
                        # the region accordingly for the overlay annotation.
                        _vis_region = region
                        if tile_param == "crop" and _af_frag is not None:
                            _vis_region = WorstRegion(
                                x=region.x - _af_frag["x"],
                                y=region.y - _af_frag["y"],
                                width=region.width,
                                height=region.height,
                                avg_distortion=region.avg_distortion,
                            )
                        _visualize_distortion_map(
                            aggregate_map,
                            _vis_region,
                            group_dir / "distortion_map_anisotropic.webp",
                            dash_color="cyan",
                        )

                    # Prepare crop-study annotation overlays
                    _annot_af: dict[str, int] | None = None
                    _annot_crop_regions: dict[int, dict] | None = None
                    if tile_param == "crop":
                        _annot_af = _af_frag if _af_frag is not None else None
                        _annot_crop_regions = {
                            lvl: entry[1]
                            for lvl, entry in crop_cache.items()
                            if entry[1] is not None
                        } or None

                    _save_annotated_original(
                        source_path,
                        region,
                        group_dir / "original_annotated.webp",
                        analysis_fragment=_annot_af,
                        crop_regions=_annot_crop_regions,
                    )

                    # ----------------------------------------------------------
                    # Phase 3: Crop and assemble grids per target value
                    # ----------------------------------------------------------
                    for target_value in target_values:
                        encoded_variants = per_value_data[target_value]
                        interpolated_qualities = per_value_qualities[target_value]

                        # If there are no real encoded images (placeholders only), skip.
                        if not encoded_variants or all(
                            enc_path is None for enc_path, _, _ in encoded_variants
                        ):
                            print(
                                f"  [{group_label}/{target_value}] "
                                f"No valid variants produced (only placeholders), skipping"
                            )
                            continue

                        target_label = f"{target_metric}_{target_value}"
                        crops_dir = work / "crops" / target_label
                        crops_dir.mkdir(parents=True, exist_ok=True)

                        original_crop = crops_dir / "original.png"
                        crop_and_zoom(
                            source_path,
                            region,
                            zoom_factor=config.zoom_factor,
                            output_path=original_crop,
                        )

                        crop_entries: list[tuple[Path, str, str]] = [
                            (original_crop, "Original", "Reference"),
                        ]
                        crop_placeholder_indices: set[int] = set()
                        variant_distmap_entries: list[tuple[np.ndarray | None, str, str]] = []

                        # When crop is the tile parameter, include it in the
                        # label so each tile shows its crop resolution.
                        label_varying = intra_res_varying
                        if tile_param == "crop" and "crop" not in label_varying:
                            label_varying = [*intra_res_varying, "crop"]

                        for enc_path, m, dm_arr in encoded_variants:
                            label = _build_label(m, label_varying)
                            safe_name = label.replace(" ", "_").replace("=", "-")

                            if enc_path is None:
                                # Placeholder: create a white spacer tile to preserve grid position
                                tile_side = config.crop_size * config.zoom_factor
                                placeholder_crop_path = crops_dir / f"placeholder_{safe_name}.png"
                                Image.new(
                                    "RGB", (tile_side, tile_side), color=(255, 255, 255)
                                ).save(placeholder_crop_path)
                                crop_entries.append((placeholder_crop_path, label, ""))
                                crop_placeholder_indices.add(len(crop_entries) - 1)
                                variant_distmap_entries.append((None, label, ""))
                                continue

                            metric_label = _build_metric_label(m)
                            crop_path = crops_dir / f"{safe_name}.png"

                            # For crop-tile studies the encoded image is at
                            # crop dimensions; map the region from original
                            # image coordinates to crop coordinates.
                            variant_region = region
                            if tile_param == "crop" and m.get("crop") is not None:
                                tile_cr_entry: tuple[Path, dict, dict | None] | None = (
                                    crop_cache.get(m["crop"])
                                )
                                if tile_cr_entry is not None:
                                    cr_dict: dict = tile_cr_entry[1]  # crop_region dict
                                    variant_region = WorstRegion(
                                        x=region.x - cr_dict["x"],
                                        y=region.y - cr_dict["y"],
                                        width=region.width,
                                        height=region.height,
                                        avg_distortion=region.avg_distortion,
                                    )

                            crop_and_zoom(
                                enc_path,
                                variant_region,
                                zoom_factor=config.zoom_factor,
                                output_path=crop_path,
                            )
                            crop_entries.append((crop_path, label, metric_label))

                            if dm_arr is not None:
                                variant_distmap_entries.append((dm_arr, label, metric_label))
                            else:
                                variant_distmap_entries.append((None, label, metric_label))

                        # Fragment comparison grid
                        value_suffix = str(target_value)
                        grid_path = group_dir / f"comparison_{value_suffix}.webp"
                        assemble_comparison_grid(
                            crop_entries,
                            grid_path,
                            max_columns=config.max_columns,
                            label_font_size=config.label_font_size,
                            figure_title=_format_figure_title(target_metric, target_value),
                            placeholder_indices=frozenset(crop_placeholder_indices),
                        )
                        output_images: list[Path] = [grid_path]
                        print(f"  [{group_label}/{target_value}] Generated: {grid_path}")

                        # Distortion map comparison grid
                        if any(arr is not None for arr, _, _ in variant_distmap_entries):
                            distmap_thumbs_dir = work / "distmap_thumbs" / target_label
                            distmap_thumbs_dir.mkdir(parents=True, exist_ok=True)
                            target_side = config.crop_size * config.zoom_factor

                            orig_thumb = distmap_thumbs_dir / "original.png"
                            with Image.open(source_path) as _src:
                                # For crop-tile studies the distortion maps cover
                                # the analysis fragment, not the full image.  Show
                                # just the fragment in the "Original" thumbnail.
                                if tile_param == "crop" and _af_frag is not None:
                                    _ox, _oy = _af_frag["x"], _af_frag["y"]
                                    _ow, _oh = _af_frag["width"], _af_frag["height"]
                                    _frag_img = _src.crop((_ox, _oy, _ox + _ow, _oy + _oh))
                                    _frag_img.convert("RGB").resize(
                                        (target_side, target_side),
                                        Image.Resampling.LANCZOS,
                                    ).save(orig_thumb)
                                else:
                                    _src.convert("RGB").resize(
                                        (target_side, target_side),
                                        Image.Resampling.LANCZOS,
                                    ).save(orig_thumb)

                            distmap_crop_entries: list[tuple[Path, str, str]] = [
                                (orig_thumb, "Original", "Reference image"),
                            ]
                            distmap_placeholder_indices: set[int] = set()
                            for idx, (dm_arr_entry, dm_label, dm_metric) in enumerate(
                                variant_distmap_entries
                            ):
                                safe_dm = dm_label.replace(" ", "_").replace("=", "-")
                                thumb_path = distmap_thumbs_dir / f"dm_{idx:03d}_{safe_dm}.png"
                                if dm_arr_entry is None:
                                    Image.new(
                                        "RGB",
                                        (target_side, target_side),
                                        color=(255, 255, 255),
                                    ).save(thumb_path)
                                    distmap_crop_entries.append((thumb_path, dm_label, ""))
                                    distmap_placeholder_indices.add(len(distmap_crop_entries) - 1)
                                else:
                                    _render_distmap_thumbnail(
                                        dm_arr_entry,
                                        target_side,
                                        thumb_path,
                                        vmax=config.distmap_vmax,
                                    )
                                    distmap_crop_entries.append((thumb_path, dm_label, dm_metric))

                            dm_grid_path = (
                                group_dir / f"distortion_map_comparison_{value_suffix}.webp"
                            )
                            assemble_comparison_grid(
                                distmap_crop_entries,
                                dm_grid_path,
                                max_columns=config.max_columns,
                                label_font_size=config.label_font_size,
                                figure_title=(
                                    f"Distortion maps \u2013 "
                                    f"{_format_figure_title(target_metric, target_value)}"
                                ),
                                placeholder_indices=frozenset(distmap_placeholder_indices),
                            )
                            output_images.append(dm_grid_path)
                            print(f"  [{group_label}/{target_value}] Generated: {dm_grid_path}")

                        target_results.append(
                            TargetComparisonResult(
                                target_metric=target_metric,
                                target_value=target_value,
                                source_image=selected_image,
                                region=region,
                                interpolated_qualities=interpolated_qualities,
                                output_images=output_images,
                            )
                        )

    print("\nComparison complete!")
    print(f"  Output directory: {output_dir}")
    total_images = sum(len(r.output_images) for r in target_results)
    print(f"  {total_images} comparison images generated")

    return ComparisonResult(
        study_id=study_id,
        target_results=target_results,
        varying_parameters=varying,
    )


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------


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
    and then scaled to ``target_size x target_size`` pixels using
    high-quality (Lanczos) down-sampling.

    Using the same ``vmin``/``vmax`` for every tile in the comparison
    grid guarantees that equal colours represent equal levels of
    distortion across all encoding variants, so structural differences
    are directly visible.

    Args:
        arr: Full-image 2-D float distortion array ``(H, W)``.
        target_size: Each output tile will be scaled to this width and
            height in pixels.
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
        badge_w = max(34, len(label) * 9 + 8)
        draw_cmd += (
            f" fill rgba(255,255,255,0.82) stroke none "
            f"roundrectangle {tx},{ty} {tx + badge_w},{ty + 18} 3,3 "
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
    grayscale-compatible: bright-yellow pixels have low distortion
    and dark-purple pixels have high distortion.  The selected
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
    *,
    analysis_fragment: dict[str, int] | None = None,
    crop_regions: dict[int, dict] | None = None,
) -> None:
    """Save a copy of the source image with the selected region annotated.

    For crop-impact studies, additional overlays are drawn:

    * The **analysis fragment** (e.g. 200×200 pixels) in green — this is
      the fixed area where quality metrics are measured.
    * Each **crop level boundary** in orange with a label so readers can
      see how much of the image is retained at every crop resolution.

    Args:
        source_path: Path to the original source image.
        region: The comparison-crop region to annotate (cyan dashes).
        output_path: Destination image path.
        analysis_fragment: ``{"x", "y", "width", "height"}`` in
            original-image coordinates for the analysis fragment overlay.
        crop_regions: Mapping of crop-level → ``{"x", "y", "width",
            "height"}`` in original-image coordinates.
    """
    # 1. Comparison-crop region (default annotation)
    _draw_annotation_on_image(source_path, region, output_path)

    # 2. Analysis fragment (green)
    if analysis_fragment is not None:
        af_region = WorstRegion(
            x=analysis_fragment["x"],
            y=analysis_fragment["y"],
            width=analysis_fragment["width"],
            height=analysis_fragment["height"],
            avg_distortion=0.0,
        )
        _draw_annotation_on_image(
            output_path,
            af_region,
            output_path,
            dash_color="lime",
            label=f"{analysis_fragment['width']}px",
        )

    # 3. Crop-level boundaries (orange, labeled)
    if crop_regions:
        for crop_level in sorted(crop_regions.keys(), reverse=True):
            cr = crop_regions[crop_level]
            cr_region = WorstRegion(
                x=cr["x"],
                y=cr["y"],
                width=cr["width"],
                height=cr["height"],
                avg_distortion=0.0,
            )
            _draw_annotation_on_image(
                output_path,
                cr_region,
                output_path,
                dash_color="orange",
                label=str(crop_level),
            )
