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

import numpy as np
from PIL import Image

from src.encoder import ImageEncoder


@dataclass
class ComparisonConfig:
    """Configuration for visual comparison generation.

    Attributes:
        crop_size: Size of the crop region in original pixels (before zoom).
        zoom_factor: Factor to scale the crop (e.g., 2 for 200% zoom).
        metric: Primary metric used to find the worst measurement.
        max_columns: Maximum number of images per row in the grid.
        label_font_size: Font size for labels in the comparison grid.
    """

    crop_size: int = 128
    zoom_factor: int = 2
    metric: str = "ssimulacra2"
    max_columns: int = 6
    label_font_size: int = 14


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
class ComparisonResult:
    """Result of the visual comparison generation.

    Attributes:
        study_id: Study identifier.
        worst_source_image: Path to the source image with worst quality.
        worst_metric_value: The worst metric score found.
        worst_format: Format of the worst measurement.
        worst_quality: Quality parameter of the worst measurement.
        region: The detected worst region coordinates.
        output_images: List of generated comparison image paths.
        varying_parameters: Parameters that vary across measurements.
    """

    study_id: str
    worst_source_image: str
    worst_metric_value: float
    worst_format: str
    worst_quality: int
    region: WorstRegion
    output_images: list[Path] = field(default_factory=list)
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
        m
        for m in measurements
        if m.get("measurement_error") is None
        and m.get(metric) is not None
    ]

    if not valid:
        msg = f"No valid measurements found for metric '{metric}'"
        raise ValueError(msg)

    if metric in higher_is_better:
        return min(valid, key=lambda m: m[metric])
    else:
        return max(valid, key=lambda m: m[metric])


def find_worst_source_image(
    measurements: list[dict],
    metric: str = "ssimulacra2",
) -> str:
    """Find the source image that produces the worst quality across all encodings.

    Computes the average metric value per source image and returns the
    source image with the worst average.

    Args:
        measurements: List of measurement dictionaries
        metric: Metric to use for comparison

    Returns:
        Path string of the source image with worst average quality

    Raises:
        ValueError: If no valid measurements exist
    """
    higher_is_better = {"ssimulacra2", "psnr", "ssim"}

    # Group metrics by source image
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

    # Compute averages
    image_avgs = {img: sum(scores) / len(scores) for img, scores in image_scores.items()}

    if metric in higher_is_better:
        return min(image_avgs, key=lambda k: image_avgs[k])
    else:
        return max(image_avgs, key=lambda k: image_avgs[k])


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
    output_map: Path,
    raw_output_map: Path | None = None,
) -> Path:
    """Generate a Butteraugli spatial distortion map.

    Uses ``butteraugli_main --distmap`` to produce a false-colour PNG
    heatmap for display, and optionally ``--rawdistmap`` to write a PFM
    file containing the actual per-pixel float distortion values (for
    use by :func:`find_worst_region`).

    Args:
        original: Path to the original reference image.
        compressed: Path to the compressed/encoded image.
        output_map: Path where the false-colour distortion map PNG will
            be written.
        raw_output_map: Optional path for the raw PFM distortion map.
            When provided, ``--rawdistmap`` is passed to
            ``butteraugli_main`` so the caller can use true float values
            for region detection instead of analysing the colour map.

    Returns:
        Path to the generated false-colour distortion map PNG.

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

        output_map.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "butteraugli_main",
            str(orig_png),
            str(comp_png),
            "--distmap",
            str(output_map),
        ]
        if raw_output_map is not None:
            raw_output_map.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--rawdistmap", str(raw_output_map)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"butteraugli_main failed: {result.stderr}"
            raise RuntimeError(msg)

    return output_map


def find_worst_region(
    distortion_map_path: Path,
    crop_size: int = 128,
) -> WorstRegion:
    """Find the region with highest distortion in a Butteraugli distortion map.

    Slides a window of ``crop_size × crop_size`` over the distortion map
    and finds the position with the highest average distortion value.

    Two input formats are supported:

    * **PFM** (``.pfm``): the raw floating-point distortion map produced
      by ``butteraugli_main --rawdistmap``.  Each pixel holds the actual
      perceptual distortion value, so this is the preferred input.
    * **PNG**: the false-colour heatmap produced by
      ``butteraugli_main --distmap``.  The luminance channel is used as
      a proxy for distortion intensity.  This is less accurate because
      the false-colour palette is non-linear.

    Args:
        distortion_map_path: Path to the distortion map (``.pfm`` or image).
        crop_size: Size of the sliding window in pixels.

    Returns:
        :class:`WorstRegion` with coordinates and average distortion score.
    """
    if distortion_map_path.suffix.lower() == ".pfm":
        arr = _read_pfm(distortion_map_path)
    else:
        with Image.open(distortion_map_path) as img:
            # Fallback: grayscale luminance as distortion proxy
            gray = img.convert("L")
            arr = np.array(gray, dtype=np.float64)

    h, w = arr.shape

    if h < crop_size or w < crop_size:
        # Image smaller than crop: use the whole image
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

    # Compute sum for each crop_size x crop_size window
    # pad integral image with zeros on top and left
    padded = np.zeros((h + 1, w + 1), dtype=np.float64)
    padded[1:, 1:] = integral

    # Window sums using the integral image
    y_max = h - crop_size + 1
    x_max = w - crop_size + 1

    window_sums = (
        padded[crop_size:h + 1, crop_size:w + 1]
        - padded[:y_max, crop_size:w + 1]
        - padded[crop_size:h + 1, :x_max]
        + padded[:y_max, :x_max]
    )

    # Find position of maximum sum
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


def crop_and_zoom(
    image_path: Path,
    region: WorstRegion,
    zoom_factor: int = 2,
    output_path: Path | None = None,
) -> Image.Image:
    """Crop a region from an image and zoom with nearest-neighbor interpolation.

    Args:
        image_path: Path to the image file
        region: Region to crop
        zoom_factor: Scale factor (e.g., 2 for 200%)
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
    max_columns: int = 6,
    label_font_size: int = 14,
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

    Args:
        source_path: Path to the source image (PNG)
        measurement: Measurement dictionary with format, quality, and
            optional encoder parameters (speed, effort, method, chroma_subsampling)
        output_dir: Directory where the encoded file will be written

    Returns:
        Path to the encoded file, or None if encoding failed
    """
    encoder = ImageEncoder(output_dir)
    fmt = measurement["format"]
    quality = measurement["quality"]
    output_name = f"encoded_{fmt}_q{quality}"

    # Add extra params to output name for uniqueness
    for param in ("chroma_subsampling", "speed", "effort", "method"):
        if measurement.get(param) is not None:
            output_name += f"_{param}{measurement[param]}"

    if fmt == "jpeg":
        result = encoder.encode_jpeg(source_path, quality, output_name=output_name)
    elif fmt == "webp":
        method = measurement.get("method") or 4
        result = encoder.encode_webp(source_path, quality, method=method, output_name=output_name)
    elif fmt == "avif":
        speed = measurement.get("speed") or 4
        chroma = measurement.get("chroma_subsampling")
        result = encoder.encode_avif(
            source_path, quality, speed=speed,
            chroma_subsampling=chroma, output_name=output_name,
        )
    elif fmt == "jxl":
        effort = measurement.get("effort") or 7
        result = encoder.encode_jxl(source_path, quality, effort=effort, output_name=output_name)
    else:
        print(f"  Warning: Unknown format '{fmt}', skipping")
        return None

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

    Args:
        source_path: Path to the source image (PNG)
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


def generate_comparison(
    quality_json_path: Path,
    output_dir: Path,
    project_root: Path,
    config: ComparisonConfig | None = None,
) -> ComparisonResult:
    """Generate visual comparison images for a study's worst-case encoding.

    This is the main entry point for the comparison feature. It:
    1. Loads quality results and finds the worst source image
    2. Obtains the encoded image (saved artifact or re-encoded on the fly)
    3. Generates a Butteraugli distortion map to locate the most degraded region
    4. Obtains all parameter variants and crops the worst region
    5. Assembles comparison grid(s) organized by varying parameters

    When the pipeline was run with ``save_worst_image=True``, encoded
    files for the worst image are used directly from disk.  Otherwise,
    images are re-encoded on the fly — only the original source images
    must be available.

    Args:
        quality_json_path: Path to the quality.json results file
        output_dir: Directory where comparison images will be saved
        project_root: Project root directory for resolving relative paths
        config: Comparison configuration (uses defaults if None)

    Returns:
        ComparisonResult with details about what was generated

    Raises:
        FileNotFoundError: If quality results or images are not found
        RuntimeError: If image processing tools fail
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

    # 2. Find the worst source image (by average metric)
    worst_source = find_worst_source_image(measurements, metric=config.metric)
    print(f"  Worst source image: {worst_source}")

    # Get measurements for this source image
    source_measurements = [
        m for m in measurements
        if m["source_image"] == worst_source
        and m.get("measurement_error") is None
        and m.get(config.metric) is not None
    ]

    if not source_measurements:
        msg = f"No valid measurements found for source image: {worst_source}"
        raise ValueError(msg)

    # Find the single worst measurement for this image (for distortion map)
    worst_m = find_worst_measurement(source_measurements, metric=config.metric)
    worst_metric_value = worst_m[config.metric]
    print(f"  Worst measurement: {worst_m['format']} q{worst_m['quality']} "
          f"({config.metric}={worst_metric_value:.2f})")

    # 3. Obtain the encoded image for the worst measurement
    source_path = project_root / worst_m["source_image"]

    if not source_path.exists():
        msg = f"Source image not found: {source_path}"
        raise FileNotFoundError(msg)

    encoded_dir = output_dir / "encoded"
    encoded_dir.mkdir(parents=True, exist_ok=True)

    # Check for pre-existing encoded files (from save_worst_image pipeline option)
    has_saved_artifacts = _resolve_encoded_path(worst_m, project_root) is not None
    if has_saved_artifacts:
        print("  Using saved encoded artifacts from pipeline")
    else:
        print("  Re-encoding on the fly (no saved artifacts found)")

    print("  Obtaining worst measurement encoded image...")
    worst_encoded = _get_or_encode(source_path, worst_m, encoded_dir, project_root)
    if worst_encoded is None:
        msg = f"Failed to obtain encoded {worst_m['format']} q{worst_m['quality']}"
        raise RuntimeError(msg)

    # 4. Generate distortion map for the worst measurement
    distmap_path = output_dir / "distortion_map.png"
    raw_distmap_path = output_dir / "distortion_map_raw.pfm"
    print("  Generating distortion map...")
    generate_distortion_map(source_path, worst_encoded, distmap_path, raw_distmap_path)

    # 5. Find the worst region using the raw PFM map when available
    # (true float values) or fall back to the colour PNG
    region_map = raw_distmap_path if raw_distmap_path.exists() else distmap_path
    region = find_worst_region(region_map, crop_size=config.crop_size)
    print(f"  Worst region: ({region.x}, {region.y}) "
          f"{region.width}x{region.height} "
          f"avg_distortion={region.avg_distortion:.2f}")

    # 6. Determine varying parameters
    varying = determine_varying_parameters(source_measurements)
    print(f"  Varying parameters: {varying}")

    # 7. Crop and zoom the original image first
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    original_crop_path = crops_dir / "original.png"
    crop_and_zoom(
        source_path, region, zoom_factor=config.zoom_factor,
        output_path=original_crop_path,
    )

    # 8. Re-encode and crop each parameter variant
    crop_entries: list[tuple[Path, str, str]] = [
        (original_crop_path, "Original", "Reference"),
    ]

    # Sort measurements for consistent ordering
    sort_keys = ["format", "quality"]
    for param in ["chroma_subsampling", "speed", "effort", "method", "resolution"]:
        if param in varying:
            sort_keys.append(param)

    sorted_measurements = sorted(
        source_measurements,
        key=lambda m: tuple(m.get(k, 0) or 0 for k in sort_keys),
    )

    print(f"  Obtaining {len(sorted_measurements)} variants...")
    for m in sorted_measurements:
        enc_path = _get_or_encode(source_path, m, encoded_dir, project_root)
        if enc_path is None:
            continue

        label = _build_label(m, varying)
        metric_label = _build_metric_label(m)

        # Create a safe filename from the label
        safe_name = label.replace(" ", "_").replace("=", "-")
        crop_path = crops_dir / f"{safe_name}.png"

        crop_and_zoom(
            enc_path, region, zoom_factor=config.zoom_factor,
            output_path=crop_path,
        )
        crop_entries.append((crop_path, label, metric_label))

    print(f"  Cropped {len(crop_entries)} images (including original)")

    # 9. Assemble comparison grids
    output_images: list[Path] = []

    # If few enough images, make a single grid
    if len(crop_entries) <= config.max_columns * 3:
        grid_path = output_dir / "comparison.png"
        assemble_comparison_grid(
            crop_entries,
            grid_path,
            max_columns=config.max_columns,
            label_font_size=config.label_font_size,
        )
        output_images.append(grid_path)
        print(f"  Generated comparison grid: {grid_path}")
    else:
        # Split by first varying parameter (typically format)
        if varying:
            split_param = varying[0]
            groups: dict[str, list[tuple[Path, str, str]]] = {}
            # Keep original in every group
            original_entry = crop_entries[0]

            for entry, m in zip(crop_entries[1:], sorted_measurements, strict=False):
                group_key = str(m.get(split_param, "unknown"))
                if group_key not in groups:
                    groups[group_key] = [original_entry]
                groups[group_key].append(entry)

            for group_name, group_entries in sorted(groups.items()):
                grid_path = output_dir / f"comparison_{split_param}_{group_name}.png"
                assemble_comparison_grid(
                    group_entries,
                    grid_path,
                    max_columns=config.max_columns,
                    label_font_size=config.label_font_size,
                )
                output_images.append(grid_path)
                print(f"  Generated comparison grid: {grid_path}")
        else:
            # No varying params, just output what we have
            grid_path = output_dir / "comparison.png"
            assemble_comparison_grid(
                crop_entries,
                grid_path,
                max_columns=config.max_columns,
                label_font_size=config.label_font_size,
            )
            output_images.append(grid_path)

    # Save the distortion map overlay with the crop region marked
    _save_annotated_distmap(distmap_path, region, output_dir / "distortion_map_annotated.png")

    print("\nComparison complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Comparison images: {len(output_images)}")

    return ComparisonResult(
        study_id=study_id,
        worst_source_image=worst_source,
        worst_metric_value=worst_metric_value,
        worst_format=worst_m["format"],
        worst_quality=worst_m["quality"],
        region=region,
        output_images=output_images,
        varying_parameters=varying,
    )


def _save_annotated_distmap(
    distmap_path: Path,
    region: WorstRegion,
    output_path: Path,
) -> None:
    """Save the distortion map with the worst region highlighted.

    Draws a high-contrast dashed rectangle over the selected crop region.
    The annotation uses a thick white solid outer border followed by a
    thin dashed black inner border so it remains visible on the
    pink/red false-colour butteraugli heatmap.

    Args:
        distmap_path: Path to the original distortion map (PNG).
        region: The worst region coordinates.
        output_path: Path where the annotated map will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x1, y1 = region.x, region.y
    x2, y2 = region.x + region.width, region.y + region.height

    # Single -draw call using MVG inline attributes:
    #   1. Solid white 5 px border  (visible on dark / saturated regions)
    #   2. Dashed black 2 px border (visible on light / pale regions)
    # stroke-dasharray applies to subsequent geometry only.
    draw_cmd = (
        f"fill none "
        f"stroke white stroke-width 5 "
        f"rectangle {x1},{y1} {x2},{y2} "
        f"stroke-dasharray 10,6 "
        f"stroke black stroke-width 2 "
        f"rectangle {x1},{y1} {x2},{y2}"
    )
    cmd = [
        "magick",
        str(distmap_path),
        "-draw",
        draw_cmd,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Non-critical, just warn
        print(f"  Warning: Could not annotate distortion map: {result.stderr}")
