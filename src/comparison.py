"""Visual comparison module for identifying and visualizing encoding artifacts.

This module automates the process of finding the worst-performing encoded
image in a study, locating the most degraded region within that image,
and generating side-by-side comparison images at 2x zoom for visual
inspection.

The workflow is:
1. Load quality measurement results
2. Find the worst measurement (lowest SSIMULACRA2 score)
3. Obtain the encoded image (from saved artifacts or by re-encoding on the fly)
4. Generate a Butteraugli distortion map to locate the most degraded region
5. Obtain all parameter variants for the worst source image
6. Crop the problematic fragment and assemble labeled comparison grids

When the pipeline was run with ``save_worst_image=True``, encoded files
for the worst image are already on disk and can be used directly.
Otherwise, images are re-encoded on the fly using the same encoder tools
as the original study.

This module requires:
- butteraugli_main (for spatial distortion maps)
- ImageMagick 7 (montage command for grid assembly)
- Pillow (for image cropping and nearest-neighbor scaling)
- Encoding tools (cjpeg, cwebp, avifenc, cjxl) matching the study formats
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from src.encoder import ImageEncoder


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
            project root) to use instead of automatic selection.  When
            set, overrides the automatic worst-image detection for all
            strategies.
    """

    crop_size: int = 128
    zoom_factor: int = 3
    metric: str = "ssimulacra2"
    max_columns: int = 4
    label_font_size: int = 22
    strategy: str = "both"
    distmap_vmax: float = 5.0
    source_image: str | None = None


@dataclass
class WorstRegion:
    """Information about the most degraded region in an image.

    Attributes:
        x: Left coordinate of the crop region.
        y: Top coordinate of the crop region.
        width: Width of the crop region in pixels.
        height: Height of the crop region in pixels.
        avg_distortion: Average distortion score in this region.
    """

    x: int
    y: int
    width: int
    height: int
    avg_distortion: float


@dataclass
class StrategyResult:
    """Result for a single image + fragment selection strategy.

    Attributes:
        strategy: The strategy used (``"average"`` or ``"variance"``).
        source_image: Path (relative to project root) of the selected source image.
        image_score: The score that determined image selection — average
            metric for ``"average"`` strategy, metric variance for ``"variance"``.
        worst_format: Format of the single worst measurement for this image.
        worst_quality: Quality parameter of the single worst measurement.
        worst_metric_value: The single worst metric score for this image.
        region: The detected worst fragment coordinates.
        output_dir: Directory where this strategy's outputs were written.
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

    Contains one :class:`StrategyResult` per strategy that was executed.
    When ``strategy="both"`` (default), there will be two entries — one
    for ``"average"`` and one for ``"variance"``.

    Attributes:
        study_id: Study identifier.
        strategies: Per-strategy results.
        varying_parameters: Parameters that vary across measurements
            (shared across strategies since they come from the same study).
    """

    study_id: str
    strategies: list[StrategyResult] = field(default_factory=list)
    varying_parameters: list[str] = field(default_factory=list)


def load_quality_results(quality_json_path: Path) -> dict:
    """Load quality measurement results from JSON file.

    Args:
        quality_json_path: Path to quality.json file

    Returns:
        Quality results dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid
    """
    if not quality_json_path.exists():
        msg = f"Quality results file not found: {quality_json_path}"
        raise FileNotFoundError(msg)

    with open(quality_json_path, encoding="utf-8") as f:
        data: dict = json.load(f)

    if "measurements" not in data:
        msg = f"No 'measurements' field in {quality_json_path}"
        raise ValueError(msg)

    return data


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


def find_worst_source_image(
    measurements: list[dict],
    metric: str = "ssimulacra2",
    strategy: str = "average",
) -> str:
    """Find the source image with the worst quality across all encodings.

    Supports two selection strategies:

    * ``"average"`` — computes the mean metric value per source image and
      returns the image with the worst average.  This identifies images
      that are consistently hard for all parameter combinations.
    * ``"variance"`` — computes the variance of metric values per source
      image and returns the image with the highest variance.  This
      identifies images whose quality is most sensitive to parameter
      changes.

    Args:
        measurements: List of measurement dictionaries.
        metric: Metric to use for comparison.
        strategy: ``"average"`` or ``"variance"``.

    Returns:
        Path string of the selected source image.

    Raises:
        ValueError: If no valid measurements exist or strategy is unknown.
    """
    if strategy not in ("average", "variance"):
        msg = f"Unknown image selection strategy: {strategy!r}. Must be 'average' or 'variance'."
        raise ValueError(msg)

    higher_is_better = {"ssimulacra2", "psnr", "ssim"}
    image_scores = _group_scores_by_image(measurements, metric)

    if strategy == "average":
        image_agg = {img: sum(s) / len(s) for img, s in image_scores.items()}
        if metric in higher_is_better:
            return min(image_agg, key=lambda k: image_agg[k])
        else:
            return max(image_agg, key=lambda k: image_agg[k])

    # strategy == "variance"
    image_var: dict[str, float] = {}
    for img, scores in image_scores.items():
        if len(scores) < 2:
            image_var[img] = 0.0
        else:
            mean = sum(scores) / len(scores)
            image_var[img] = sum((s - mean) ** 2 for s in scores) / len(scores)
    return max(image_var, key=lambda k: image_var[k])


def _find_worst_original_image(
    measurements: list[dict],
    metric: str = "ssimulacra2",
    strategy: str = "average",
) -> str:
    """Find the worst original image across all resolutions and encodings.

    Like :func:`find_worst_source_image` but groups by ``original_image``
    instead of ``source_image``.  This is used for resolution studies
    where each ``(image, resolution)`` pair has a distinct ``source_image``
    but shares the same ``original_image``.

    Args:
        measurements: List of measurement dictionaries.
        metric: Metric to use for comparison.
        strategy: ``"average"`` or ``"variance"``.

    Returns:
        The ``original_image`` path string of the selected image.

    Raises:
        ValueError: If no valid measurements exist or strategy is unknown.
    """
    if strategy not in ("average", "variance"):
        msg = f"Unknown strategy: {strategy!r}. Must be 'average' or 'variance'."
        raise ValueError(msg)

    higher_is_better = {"ssimulacra2", "psnr", "ssim"}

    # Group scores by original_image
    image_scores: dict[str, list[float]] = {}
    for m in measurements:
        if m.get("measurement_error") is not None or m.get(metric) is None:
            continue
        key = m.get("original_image", m["source_image"])
        image_scores.setdefault(key, []).append(m[metric])

    if not image_scores:
        msg = f"No valid measurements with metric '{metric}' found"
        raise ValueError(msg)

    if strategy == "average":
        image_agg = {img: sum(s) / len(s) for img, s in image_scores.items()}
        if metric in higher_is_better:
            return min(image_agg, key=lambda k: image_agg[k])
        else:
            return max(image_agg, key=lambda k: image_agg[k])

    # strategy == "variance"
    image_var: dict[str, float] = {}
    for img, scores in image_scores.items():
        if len(scores) < 2:
            image_var[img] = 0.0
        else:
            mean = sum(scores) / len(scores)
            image_var[img] = sum((s - mean) ** 2 for s in scores) / len(scores)
    return max(image_var, key=lambda k: image_var[k])


def get_worst_image_score(
    measurements: list[dict],
    source_image: str,
    metric: str = "ssimulacra2",
    strategy: str = "average",
) -> float:
    """Compute the image-level score for a given source image and strategy.

    Args:
        measurements: List of measurement dictionaries.
        source_image: The source image path to compute the score for.
        metric: Metric name.
        strategy: ``"average"`` or ``"variance"``.

    Returns:
        The computed score (mean or variance of the metric values for
        this image).
    """
    image_scores = _group_scores_by_image(measurements, metric)
    scores = image_scores.get(source_image, [])
    if not scores:
        return 0.0

    mean = sum(scores) / len(scores)
    if strategy == "average":
        return mean

    if len(scores) < 2:
        return 0.0
    return sum((s - mean) ** 2 for s in scores) / len(scores)


def _to_png(image_path: Path, output_path: Path) -> None:
    """Convert an image to PNG format for measurement tools.

    Handles JPEG XL and AVIF via external decoders, other formats via Pillow.

    Args:
        image_path: Path to the source image
        output_path: Path where PNG will be written

    Raises:
        OSError: If conversion fails
    """
    if image_path.suffix.lower() in (".jxl", ".jpegxl"):
        try:
            cmd = ["djxl", str(image_path), str(output_path)]
            subprocess.run(cmd, capture_output=True, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msg = f"Failed to decode JXL file {image_path}: {e}"
            raise OSError(msg) from e

    if image_path.suffix.lower() == ".avif":
        try:
            cmd = ["avifdec", str(image_path), str(output_path)]
            subprocess.run(cmd, capture_output=True, check=True)
            return
        except FileNotFoundError:
            pass

    try:
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                converted = img.convert("RGB")
                converted.save(output_path, format="PNG")
            else:
                img.save(output_path, format="PNG")
    except Exception as e:
        msg = f"Failed to convert {image_path} to PNG: {e}"
        raise OSError(msg) from e


def _read_pfm(pfm_path: Path) -> np.ndarray:
    """Read a PFM (Portable Float Map) file into a 2-D float64 numpy array.

    Handles both grayscale (``Pf``) and colour (``PF``) PFM files.  For
    colour PFM the maximum across the three channels is returned so that
    the highest distortion value at each pixel is preserved.

    PFM stores rows bottom-to-top; this function flips the result to the
    conventional top-to-bottom orientation used everywhere else.

    Args:
        pfm_path: Path to the ``.pfm`` file.

    Returns:
        2-D array of shape ``(H, W)`` with distortion values.

    Raises:
        ValueError: If the file is not a valid PFM.
    """
    import struct

    with open(pfm_path, "rb") as fh:
        magic = fh.readline().decode("ascii").strip()
        if magic not in ("PF", "Pf"):
            msg = f"Not a PFM file (magic={magic!r}): {pfm_path}"
            raise ValueError(msg)
        is_color = magic == "PF"

        dims = fh.readline().decode("ascii").strip().split()
        width, height = int(dims[0]), int(dims[1])

        scale = float(fh.readline().decode("ascii").strip())
        little_endian = scale < 0

        channels = 3 if is_color else 1
        n_floats = width * height * channels
        raw = fh.read(n_floats * 4)

    endian = "<" if little_endian else ">"
    data = np.array(struct.unpack(f"{endian}{n_floats}f", raw), dtype=np.float64)

    if is_color:
        data = data.reshape((height, width, 3))
        data = data.max(axis=2)  # take worst distortion per pixel
    else:
        data = data.reshape((height, width))

    # PFM rows are stored bottom-to-top; flip to standard orientation
    return np.flipud(data)


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
            _to_png(original, orig_png)
        if compressed.suffix.lower() != ".png":
            comp_png = tmpdir_path / "compressed.png"
            _to_png(compressed, comp_png)

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


def _find_worst_region_in_array(
    arr: np.ndarray,
    crop_size: int,
) -> WorstRegion:
    """Find the worst region in a 2-D distortion value array.

    Core sliding-window computation shared by :func:`find_worst_region` and
    :func:`compute_aggregate_distortion_maps`.  Works with any float array,
    whether it comes from a raw PFM file, an averaged map, or a variance map.

    Args:
        arr: 2-D float array of shape ``(H, W)`` with per-pixel distortion
            values.  Higher values mean more distortion.
        crop_size: Side length of the square sliding window in pixels.

    Returns:
        :class:`WorstRegion` for the window position with the highest sum.
    """
    h, w = arr.shape

    if h < crop_size or w < crop_size:
        actual_h = min(h, crop_size)
        actual_w = min(w, crop_size)
        return WorstRegion(
            x=0,
            y=0,
            width=actual_w,
            height=actual_h,
            avg_distortion=float(np.mean(arr)),
        )

    # Use integral image (summed area table) for efficient sliding window
    integral = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    padded = np.zeros((h + 1, w + 1), dtype=np.float64)
    padded[1:, 1:] = integral

    y_max = h - crop_size + 1
    x_max = w - crop_size + 1

    window_sums = (
        padded[crop_size : h + 1, crop_size : w + 1]
        - padded[:y_max, crop_size : w + 1]
        - padded[crop_size : h + 1, :x_max]
        + padded[:y_max, :x_max]
    )

    max_idx = np.argmax(window_sums)
    best_y, best_x = np.unravel_index(max_idx, window_sums.shape)
    avg_distortion = float(window_sums[best_y, best_x] / (crop_size * crop_size))

    return WorstRegion(
        x=int(best_x),
        y=int(best_y),
        width=crop_size,
        height=crop_size,
        avg_distortion=avg_distortion,
    )


def find_worst_region(
    pfm_path: Path,
    crop_size: int = 128,
) -> WorstRegion:
    """Find the region with the highest distortion in a raw Butteraugli PFM file.

    Reads the per-pixel float distortion values from a PFM file produced by
    ``butteraugli_main --rawdistmap`` and delegates the sliding-window search
    to :func:`_find_worst_region_in_array`.

    Args:
        pfm_path: Path to a ``.pfm`` raw distortion map.
        crop_size: Side length of the square sliding window in pixels.

    Returns:
        :class:`WorstRegion` with coordinates and average distortion score.
    """
    arr = _read_pfm(pfm_path)
    return _find_worst_region_in_array(arr, crop_size)


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
            _to_png(image_path, src)
        elif image_path.suffix.lower() == ".avif":
            try:
                src = Path(tmpdir) / "decoded.png"
                _to_png(image_path, src)
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


def compute_aggregate_distortion_maps(
    source_path: Path,
    measurements: list[dict],
    output_dir: Path,
    project_root: Path,
    encoded_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, dict]]]:
    """Compute per-pixel average and variance distortion across all variants.

    For each measurement in ``measurements`` this function:

    1. Obtains the encoded image (saved artifact or re-encodes on the fly).
    2. Runs ``butteraugli_main --rawdistmap`` to get per-pixel float
       distortion values.
    3. Stacks all resulting maps and returns the mean and variance arrays.

    The **average map** captures regions that are consistently difficult
    for every parameter combination.  The **variance map** captures
    regions where quality varies most across parameter combinations — a
    high-variance pixel is encoded well by some settings but badly by
    others.

    Args:
        source_path: Path to the source (original) image.
        measurements: List of measurement dicts for the source image.
        output_dir: Directory where per-variant distortion maps will be
            written (under a ``distmaps/`` subdirectory).
        project_root: Project root for resolving saved artifact paths.
        encoded_dir: Directory for re-encoded files produced on the fly.

    Returns:
        ``(avg_map, var_map)`` — two float64 arrays of shape ``(H, W)``.

    Raises:
        RuntimeError: If no distortion maps could be produced for any variant.
    """
    pfms_dir = output_dir / "pfms"
    pfms_dir.mkdir(parents=True, exist_ok=True)

    arrays: list[np.ndarray] = []
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

        raw_pfm = pfms_dir / f"distmap_{suffix}.pfm"
        try:
            generate_distortion_map(source_path, enc_path, raw_pfm)
        except RuntimeError as exc:
            print(f"    Warning: distortion map failed for {fmt} q{quality}: {exc}")
            continue

        try:
            arr = _read_pfm(raw_pfm)
        except ValueError as exc:
            print(f"    Warning: could not read PFM for {fmt} q{quality}: {exc}")
            continue

        arrays.append(arr)
        variant_pairs.append((arr, m))

    if not arrays:
        msg = "No distortion maps could be computed for any variant"
        raise RuntimeError(msg)

    stacked = np.stack(arrays, axis=0)  # (N, H, W)
    return stacked.mean(axis=0), stacked.var(axis=0), variant_pairs


def generate_comparison(
    quality_json_path: Path,
    output_dir: Path,
    project_root: Path,
    config: ComparisonConfig | None = None,
) -> ComparisonResult:
    """Generate visual comparison images for a study's worst-case encoding.

    This is the main entry point for the comparison feature.  For each
    requested strategy (``"average"``, ``"variance"``, or ``"both"``):

    1. Select the worst source image using the strategy's image-selection
       logic (or use the explicitly provided ``source_image``).
    2. Compute Butteraugli distortion maps across all parameter variants.
    3. Locate the worst image fragment using the same strategy.
    4. Crop the fragment from every variant and assemble comparison grids.

    Each strategy's outputs are placed in a dedicated subdirectory
    (``<output_dir>/average/`` or ``<output_dir>/variance/``).

    When the study includes a ``resolution`` sweep, measurements are
    grouped by resolution.  Each resolution gets its own distortion maps,
    fragment selection, comparison grids, and annotated originals — all
    placed in per-resolution subdirectories (e.g.
    ``<output_dir>/average/r720/``).

    When the pipeline was run with ``save_worst_image=True``, encoded
    files for the worst image are used directly from disk.  Otherwise,
    images are re-encoded on the fly — only the original source images
    must be available.

    Args:
        quality_json_path: Path to the quality.json results file.
        output_dir: Directory where comparison images will be saved.
        project_root: Project root directory for resolving relative paths.
        config: Comparison configuration (uses defaults if ``None``).

    Returns:
        :class:`ComparisonResult` with per-strategy details.

    Raises:
        FileNotFoundError: If quality results or images are not found.
        RuntimeError: If image processing tools fail.
    """
    if config is None:
        config = ComparisonConfig()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which strategies to run
    if config.strategy == "both":
        strategies = ["average", "variance"]
    elif config.strategy in ("average", "variance"):
        strategies = [config.strategy]
    else:
        msg = f"Unknown strategy {config.strategy!r}. Must be 'average', 'variance', or 'both'."
        raise ValueError(msg)

    # 1. Load quality results
    data = load_quality_results(quality_json_path)
    measurements = data["measurements"]
    study_id = data.get("study_id", "unknown")

    print(f"Generating visual comparison for study: {study_id}")
    print(f"  Total measurements: {len(measurements)}")
    print(f"  Strategies: {', '.join(strategies)}")

    # Determine varying parameters
    varying = determine_varying_parameters(measurements)

    # Detect whether resolution varies — if so, we must process per-resolution
    resolution_varies = "resolution" in varying
    if resolution_varies:
        # Collect unique resolution values
        unique_resolutions: list[int | None] = sorted(
            {m.get("resolution") for m in measurements if m.get("resolution") is not None}
        )
        print(f"  Resolution varies: {len(unique_resolutions)} levels "
              f"({', '.join(f'r{r}' for r in unique_resolutions if r is not None)})")
        # Varying params for labels within a resolution group (exclude resolution)
        intra_res_varying = [p for p in varying if p != "resolution"]
    else:
        unique_resolutions = [None]
        intra_res_varying = varying

    # Determine which original images to process per strategy
    # When resolution varies, we select the worst *original* image by grouping
    # across all resolutions.  For non-resolution studies the existing logic
    # selecting by source_image works fine.
    image_per_strategy: dict[str, str] = {}
    for strat in strategies:
        if config.source_image is not None:
            image_per_strategy[strat] = config.source_image
        else:
            if resolution_varies:
                # Group by original_image across all resolutions
                selected = _find_worst_original_image(
                    measurements, metric=config.metric, strategy=strat,
                )
            else:
                selected = find_worst_source_image(
                    measurements, metric=config.metric, strategy=strat,
                )
            image_per_strategy[strat] = selected
        print(f"  [{strat}] Selected image: {image_per_strategy[strat]}")

    strategy_results: list[StrategyResult] = []

    with tempfile.TemporaryDirectory() as _work:
        work = Path(_work)

        for strat in strategies:
            # Cache for preprocessed images: resolution → path.
            # Kept per-strategy so that different strategies selecting
            # different source images never share a cached resize.
            preprocess_cache: dict[int, Path] = {}
            img_key = image_per_strategy[strat]

            # Resolve the original image path
            if resolution_varies:
                # img_key is an original_image path
                original_path = project_root / img_key
            else:
                original_path = project_root / img_key

            if not original_path.exists():
                msg = f"Source image not found: {original_path}"
                raise FileNotFoundError(msg)

            strat_dir = output_dir / strat
            strat_dir.mkdir(parents=True, exist_ok=True)

            all_output_images: list[Path] = []
            first_region = None
            first_worst_m = None
            first_image_score = None

            for resolution in unique_resolutions:
                # Get the right source for this resolution
                source_path = _resolve_source_for_resolution(
                    original_path, resolution, preprocess_cache, work / "prep",
                )

                # Filter measurements for this image and resolution
                if resolution_varies:
                    source_measurements = [
                        m for m in measurements
                        if m["original_image"] == img_key
                        and m.get("resolution") == resolution
                        and m.get("measurement_error") is None
                        and m.get(config.metric) is not None
                    ]
                else:
                    source_measurements = [
                        m for m in measurements
                        if m["source_image"] == img_key
                        and m.get("measurement_error") is None
                        and m.get(config.metric) is not None
                    ]

                if not source_measurements:
                    continue

                res_label = f"r{resolution}" if resolution is not None else None
                res_dir = strat_dir / res_label if res_label else strat_dir

                encoded_dir = work / "encoded" / (res_label or "original") / Path(img_key).stem
                encoded_dir.mkdir(parents=True, exist_ok=True)

                print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                      f"Computing distortion maps for {Path(img_key).name} "
                      f"({len(source_measurements)} variants)...")

                avg_map, var_map, variant_pairs = compute_aggregate_distortion_maps(
                    source_path,
                    source_measurements,
                    work / "distmaps" / (res_label or "original") / Path(img_key).stem,
                    project_root,
                    encoded_dir,
                )

                # Select region using the strategy
                avg_region = _find_worst_region_in_array(avg_map, crop_size=config.crop_size)
                var_region = _find_worst_region_in_array(var_map, crop_size=config.crop_size)
                region = avg_region if strat == "average" else var_region

                print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                      f"Region: ({region.x}, {region.y}) "
                      f"{region.width}x{region.height} score={region.avg_distortion:.4f}")

                # Track first resolution's data for per-strategy summary
                if first_region is None:
                    first_region = region
                    first_worst_m = find_worst_measurement(source_measurements, metric=config.metric)
                    if resolution_varies:
                        # Compute image-level score across all resolutions
                        all_img_measurements = [
                            m for m in measurements
                            if m["original_image"] == img_key
                            and m.get("measurement_error") is None
                            and m.get(config.metric) is not None
                        ]
                        scores = [m[config.metric] for m in all_img_measurements]
                        mean = sum(scores) / len(scores) if scores else 0.0
                        if strat == "average":
                            first_image_score = mean
                        elif len(scores) >= 2:
                            first_image_score = sum((s - mean) ** 2 for s in scores) / len(scores)
                        else:
                            first_image_score = 0.0
                    else:
                        first_image_score = get_worst_image_score(
                            measurements, img_key, metric=config.metric, strategy=strat,
                        )

                # Build precomputed distmap lookup
                precomputed_distmaps: dict[int, np.ndarray] = {
                    id(m_dict): arr for arr, m_dict in variant_pairs
                }

                # Set up per-resolution working directories
                crops_dir = work / "crops" / strat / (res_label or "original")
                crops_dir.mkdir(parents=True)

                # Crop original
                original_crop_path = crops_dir / "original.png"
                crop_and_zoom(
                    source_path,
                    region,
                    zoom_factor=config.zoom_factor,
                    output_path=original_crop_path,
                )

                # Build crop entries
                crop_entries: list[tuple[Path, str, str]] = [
                    (original_crop_path, "Original", "Reference"),
                ]

                # Sort measurements — use intra-resolution varying params
                sort_keys = ["format", "quality"]
                for param in ["chroma_subsampling", "speed", "effort", "method"]:
                    if param in intra_res_varying:
                        sort_keys.append(param)

                sorted_measurements = sorted(
                    source_measurements,
                    key=lambda m: tuple(m.get(k, 0) or 0 for k in sort_keys),
                )

                variant_distmap_entries: list[tuple[np.ndarray, str, str]] = []

                print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                      f"Obtaining {len(sorted_measurements)} variants...")
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
                        dm_arr = precomputed_distmaps[id(m)]
                        variant_distmap_entries.append((dm_arr, label, metric_label))
                    else:
                        variant_pfm = work / f"dm_{strat}_{res_label or 'orig'}_{safe_name}.pfm"
                        try:
                            generate_distortion_map(source_path, enc_path, variant_pfm)
                            dm_arr = _read_pfm(variant_pfm)
                            variant_distmap_entries.append((dm_arr, label, metric_label))
                        except (RuntimeError, ValueError) as exc:
                            print(f"    Warning: distortion map for {label} failed: {exc}")

                print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                      f"Cropped {len(crop_entries)} images (including original)")

                # Assemble comparison grids
                output_images: list[Path] = []
                res_dir.mkdir(parents=True, exist_ok=True)

                if len(crop_entries) <= config.max_columns * 3:
                    grid_path = res_dir / "comparison.webp"
                    assemble_comparison_grid(
                        crop_entries,
                        grid_path,
                        max_columns=config.max_columns,
                        label_font_size=config.label_font_size,
                    )
                    output_images.append(grid_path)
                    print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                          f"Generated comparison grid: {grid_path}")
                else:
                    if intra_res_varying:
                        split_param = intra_res_varying[0]
                        groups: dict[str, list[tuple[Path, str, str]]] = {}
                        original_entry = crop_entries[0]

                        for entry, m in zip(crop_entries[1:], sorted_measurements, strict=False):
                            group_key = str(m.get(split_param, "unknown"))
                            if group_key not in groups:
                                groups[group_key] = [original_entry]
                            groups[group_key].append(entry)

                        for group_name, group_entries in sorted(groups.items()):
                            grid_path = res_dir / f"comparison_{split_param}_{group_name}.webp"
                            assemble_comparison_grid(
                                group_entries,
                                grid_path,
                                max_columns=config.max_columns,
                                label_font_size=config.label_font_size,
                            )
                            output_images.append(grid_path)
                            print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                                  f"Generated comparison grid: {grid_path}")
                    else:
                        grid_path = res_dir / "comparison.webp"
                        assemble_comparison_grid(
                            crop_entries,
                            grid_path,
                            max_columns=config.max_columns,
                            label_font_size=config.label_font_size,
                        )
                        output_images.append(grid_path)

                # Distortion map comparison grid
                if variant_distmap_entries:
                    distmap_thumbs_dir = work / "distmap_thumbs" / strat / (res_label or "original")
                    distmap_thumbs_dir.mkdir(parents=True)
                    target_side = config.crop_size * config.zoom_factor

                    orig_thumb = distmap_thumbs_dir / "original.png"
                    with Image.open(source_path) as _src:
                        _src.convert("RGB").resize(
                            (target_side, target_side), Image.Resampling.LANCZOS,
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

                    if len(distmap_crop_entries) <= config.max_columns * 3:
                        dm_grid_path = res_dir / "distortion_map_comparison.webp"
                        assemble_comparison_grid(
                            distmap_crop_entries,
                            dm_grid_path,
                            max_columns=config.max_columns,
                            label_font_size=config.label_font_size,
                        )
                        output_images.append(dm_grid_path)
                        print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                              f"Generated distortion map comparison grid: {dm_grid_path}")
                    else:
                        if intra_res_varying:
                            split_param = intra_res_varying[0]
                            dm_groups: dict[str, list[tuple[Path, str, str]]] = {}
                            dm_orig_entry = distmap_crop_entries[0]
                            for dm_entry, dm_m in zip(
                                distmap_crop_entries[1:], sorted_measurements, strict=False,
                            ):
                                group_key = str(dm_m.get(split_param, "unknown"))
                                if group_key not in dm_groups:
                                    dm_groups[group_key] = [dm_orig_entry]
                                dm_groups[group_key].append(dm_entry)
                            for group_name, group_entries in sorted(dm_groups.items()):
                                dm_grid_path = res_dir / (
                                    f"distortion_map_comparison_{split_param}_{group_name}.webp"
                                )
                                assemble_comparison_grid(
                                    group_entries,
                                    dm_grid_path,
                                    max_columns=config.max_columns,
                                    label_font_size=config.label_font_size,
                                )
                                output_images.append(dm_grid_path)
                                print(f"  [{strat}]{f' [{res_label}]' if res_label else ''} "
                                      f"Generated distortion map grid: {dm_grid_path}")
                        else:
                            dm_grid_path = res_dir / "distortion_map_comparison.webp"
                            assemble_comparison_grid(
                                distmap_crop_entries,
                                dm_grid_path,
                                max_columns=config.max_columns,
                                label_font_size=config.label_font_size,
                            )
                            output_images.append(dm_grid_path)

                # Distortion map visualizations
                _visualize_distortion_map(
                    avg_map if strat == "average" else var_map,
                    region,
                    res_dir / f"distortion_map_{strat}.webp",
                    dash_color="cyan" if strat == "average" else "orange",
                )
                _save_annotated_original(
                    source_path,
                    region,
                    res_dir / "original_annotated.webp",
                )

                all_output_images.extend(output_images)

            # Build strategy result from first resolution (or only resolution)
            worst_m = first_worst_m or find_worst_measurement(
                [m for m in measurements
                 if (m["original_image"] if resolution_varies else m["source_image"]) == img_key
                 and m.get("measurement_error") is None
                 and m.get(config.metric) is not None],
                metric=config.metric,
            )
            worst_metric_value = worst_m[config.metric]

            image_score = first_image_score if first_image_score is not None else 0.0
            region_for_result = first_region or WorstRegion(0, 0, config.crop_size, config.crop_size, 0.0)

            strategy_results.append(StrategyResult(
                strategy=strat,
                source_image=img_key,
                image_score=image_score,
                worst_format=worst_m["format"],
                worst_quality=worst_m["quality"],
                worst_metric_value=worst_metric_value,
                region=region_for_result,
                output_dir=strat_dir,
                output_images=all_output_images,
            ))

    print("\nComparison complete!")
    print(f"  Output directory: {output_dir}")
    for sr in strategy_results:
        print(f"  [{sr.strategy}] {len(sr.output_images)} comparison images "
              f"in {sr.output_dir.name}/")

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


def _save_annotated_original_dual(
    source_path: Path,
    avg_region: WorstRegion,
    var_region: WorstRegion,
    output_path: Path,
) -> None:
    """Save a copy of the source image annotated with both strategy regions.

    Draws two overlapping annotation boxes with distinct colours and labels
    so the viewer can compare both selection strategies at a glance:

    * **cyan / dashed** — region selected by the *average* distortion strategy.
    * **orange / dashed** — region selected by the *variance* strategy.

    Each box carries a small coloured text badge (``"avg"`` / ``"var"``)
    at its top-left corner so the two annotations remain identifiable even
    when the regions overlap.

    Args:
        source_path: Path to the original source image.
        avg_region: Region chosen by the average-distortion strategy.
        var_region: Region chosen by the variance strategy.
        output_path: Destination image path (PNG or lossless WebP by suffix).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _region_draw_cmd(reg: WorstRegion, color: str, text: str) -> str:
        x1, y1 = reg.x, reg.y
        x2, y2 = reg.x + reg.width, reg.y + reg.height
        tx, ty = x1 + 2, y1 + 2
        return (
            f"push graphic-context "
            f"fill none stroke white stroke-width 5 "
            f"rectangle {x1},{y1} {x2},{y2} "
            f"stroke-dasharray 10,6 stroke {color} stroke-width 2 "
            f"rectangle {x1},{y1} {x2},{y2} "
            f"fill rgba(255,255,255,0.82) stroke none "
            f"roundrectangle {tx},{ty} {tx + 34},{ty + 18} 3,3 "
            f"fill {color} font DejaVu-Sans font-size 13 stroke none "
            f"text {tx + 4},{ty + 14} '{text}' "
            f"pop graphic-context"
        )

    draw_cmd = (
        _region_draw_cmd(avg_region, "cyan", "avg")
        + " "
        + _region_draw_cmd(var_region, "orange", "var")
    )
    extra_args: list[str] = []
    if output_path.suffix.lower() == ".webp":
        extra_args = ["-define", "webp:lossless=true"]
    cmd = ["magick", str(source_path), "-draw", draw_cmd, *extra_args, str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: Could not annotate image: {result.stderr}")


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
