"""Merged encode+measure pipeline with time-budget support.

This module provides a unified pipeline that processes images with a
worker-per-image architecture, encoding and measuring quality in a single
pass. This eliminates the need to store intermediate encoded files on disk
and allows time-budget-based processing where the pipeline processes as
many images as possible within a given time constraint.

Architecture:
- Each worker processes one complete image at a time (all encoding tasks
  sequentially)
- Workers pull the next image from a queue when they finish
- This keeps all workers fully utilized throughout the pipeline
- Memory-intensive operations are naturally staggered across workers,
  reducing peak memory usage

Key advantages over the separate encode → measure workflow:
- **Time-budget control**: Set a wall-clock time limit instead of guessing
  how many images to process. The pipeline processes as many images as
  possible within the budget.
- **Full worker utilization**: Workers always have work available, no idle
  time waiting for other tasks to complete.
- **Reduced peak memory**: Tasks are staggered across workers rather than
  synchronized, preventing memory spikes from parallel execution of
  memory-intensive tools.
- **Reduced disk IO**: Encoded files are written to temporary storage
  and cleaned up after measurement. Optional ``save_artifacts`` flag
  persists them to disk.
- **Per-image error isolation**: All operations for one image are grouped
  within a single worker. If encoding or measurement fails, the worker
  logs the error and moves to the next image.
"""

import multiprocessing
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.dataset import DatasetFetcher
from src.encoder import ImageEncoder, get_encoder_version
from src.preprocessing import ImagePreprocessor
import numpy as np

from src.quality import (
    QualityMeasurer,
    QualityRecord,
    QualityResults,
    find_worst_region_in_array,
    get_measurement_tool_version,
    read_pfm,
)
from src.study import EncoderConfig, StudyConfig

# Default crop size used for worst-fragment detection during measurement.
DEFAULT_CROP_SIZE: int = 128

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rel(path: Path, root: Path) -> str:
    """Make *path* relative to *root*, falling back to ``str(path)``."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _format_duration(seconds: float) -> str:
    """Format seconds as ``1h 02m 03s`` / ``5m 03s`` / ``12s``."""
    if seconds < 0:
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def parse_time_budget(value: str) -> float:
    """Parse a human-readable time budget string into seconds.

    Accepted formats:

    - Plain number: interpreted as seconds (``"3600"`` → 3600.0)
    - Duration suffixes: ``"1h"``, ``"30m"``, ``"90s"``, ``"1h30m"``, ``"2h15m30s"``

    Args:
        value: Time budget string.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the format cannot be parsed.
    """
    # Plain number → seconds
    try:
        return float(value)
    except ValueError:
        pass

    pattern = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    match = re.fullmatch(pattern, value.strip())
    if not match or not any(match.groups()):
        msg = (
            f"Invalid time budget format: {value!r}. "
            f"Use seconds (3600) or duration (1h, 30m, 1h30m)"
        )
        raise ValueError(msg)

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    secs = int(match.group(3) or 0)
    return float(hours * 3600 + minutes * 60 + secs)


# ---------------------------------------------------------------------------
# Output-name builder (mirrors study._build_output_name)
# ---------------------------------------------------------------------------


def _build_output_name(
    stem: str,
    quality: int,
    chroma_subsampling: str | None = None,
    speed: int | None = None,
    effort: int | None = None,
    method: int | None = None,
    resolution: int | None = None,
) -> str:
    """Build a descriptive output filename for an encoding task."""
    parts = [stem, f"q{quality}"]
    if chroma_subsampling is not None:
        parts.append(chroma_subsampling)
    if speed is not None:
        parts.append(f"s{speed}")
    if effort is not None:
        parts.append(f"e{effort}")
    if method is not None:
        parts.append(f"m{method}")
    if resolution is not None:
        parts.append(f"r{resolution}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Image-level processing (top-level for multiprocessing)
# ---------------------------------------------------------------------------


def _process_image(
    image_path_str: str,
    config_dict: dict[str, Any],
    project_root_str: str,
    save_artifacts: bool,
    save_artifact_dir_str: str | None = None,
) -> list[QualityRecord]:
    """Process a single image through all encoder configurations.

    This function is the unit of work for each worker. It processes
    all encoding tasks for one image sequentially, which:
    - Keeps workers fully utilized
    - Naturally staggers memory-intensive operations across workers
    - Provides better cache locality

    Resolution is now part of the per-encoder Cartesian product.
    Preprocessing (resizing) is handled inline with an in-worker cache
    so each resolution is only computed once per image.

    Args:
        image_path_str: Absolute path to the source image.
        config_dict: Serializable dict with study config data.
        project_root_str: Project root path (for pickling).
        save_artifacts: Whether to save encoded files.
        save_artifact_dir_str: Override the artifact directory. When
            set, encoded files are saved here instead of the default
            ``data/encoded/<study_id>/`` path.  This is used by the
            ``save_worst_image`` pipeline option to route each image's
            files to a per-image staging directory.

    Returns:
        ``(records, fragment_info)`` — a list of :class:`QualityRecord`
        objects for all tasks, and a dict with worst-fragment metadata
        per strategy per resolution:
        ``{strategy: {resolution_key: {x, y, width, height, avg_distortion}}}``.
        Resolution keys are ``int`` or ``None`` (original resolution).
    """
    import tempfile

    from src.preprocessing import ImagePreprocessor
    from src.study import EncoderConfig

    image_path = Path(image_path_str)
    project_root = Path(project_root_str)
    study_id = config_dict["id"]

    # Reconstruct encoder configs
    encoders = [EncoderConfig(**enc_dict) for enc_dict in config_dict["encoders"]]

    # Determine artifact save directory
    if save_artifact_dir_str is not None:
        save_artifact_dir: Path | None = Path(save_artifact_dir_str)
    elif save_artifacts:
        save_artifact_dir = project_root / "data" / "encoded" / study_id
    else:
        save_artifact_dir = None

    all_records: list[QualityRecord] = []

    # In-worker preprocessing cache: resolution → preprocessed path
    preprocessed_cache: dict[int, Path] = {}

    # Running distortion-map accumulators per resolution group.
    # Using running sums instead of stacking all arrays keeps memory
    # constant regardless of the number of encoding variants.
    distmap_sums: dict[int | None, np.ndarray] = {}
    distmap_sumsq: dict[int | None, np.ndarray] = {}
    distmap_counts: dict[int | None, int] = {}

    with tempfile.TemporaryDirectory() as prep_tmpdir:
        def _get_source(resolution: int | None) -> Path:
            """Get source image, resizing if needed (cached)."""
            if resolution is None:
                return image_path
            if resolution in preprocessed_cache:
                return preprocessed_cache[resolution]
            res_dir = Path(prep_tmpdir) / f"r{resolution}"
            preprocessor = ImagePreprocessor(res_dir)
            output_name = f"{image_path.stem}_r{resolution}.png"
            resized = preprocessor.resize_image(
                image_path,
                target_size=(resolution, resolution),
                output_name=output_name,
                keep_aspect_ratio=True,
            )
            preprocessed_cache[resolution] = resized
            return resized

        # Process all encoders — resolution is part of each encoder's sweep
        for enc in encoders:
            tasks = _expand_encoder_tasks(
                source_image=image_path,
                original_image=image_path,
                enc=enc,
                save_artifact_dir=save_artifact_dir,
                study_id=study_id,
            )

            # Execute tasks sequentially within this worker
            for task_kw in tasks:
                # Resolve the actual source image for this task's resolution
                resolution = task_kw["resolution"]
                actual_source = _get_source(resolution)
                task_kw["source_image"] = str(actual_source)

                record, distmap_arr = _encode_and_measure(
                    project_root_str=project_root_str,
                    **task_kw,
                )
                all_records.append(record)

                # Accumulate distortion map into running sums
                if distmap_arr is not None:
                    if resolution not in distmap_sums:
                        distmap_sums[resolution] = np.zeros_like(distmap_arr)
                        distmap_sumsq[resolution] = np.zeros_like(distmap_arr)
                        distmap_counts[resolution] = 0
                    distmap_sums[resolution] += distmap_arr
                    distmap_sumsq[resolution] += distmap_arr ** 2
                    distmap_counts[resolution] += 1

    # --- Compute worst fragments per resolution per strategy ----------------
    fragment_info: dict[str, dict] = {}
    for res, n in distmap_counts.items():
        if n == 0:
            continue
        avg_map = distmap_sums[res] / n
        # Numerically stable variance: E[X^2] - E[X]^2, clamped to >= 0
        var_map = np.maximum(distmap_sumsq[res] / n - avg_map ** 2, 0.0)

        avg_region = find_worst_region_in_array(avg_map, crop_size=DEFAULT_CROP_SIZE)
        var_region = find_worst_region_in_array(var_map, crop_size=DEFAULT_CROP_SIZE)

        # Use the raw resolution key (int or None) — it's JSON-serialisable
        res_key: int | None = res
        fragment_info.setdefault("average", {})[res_key] = {
            "x": avg_region.x,
            "y": avg_region.y,
            "width": avg_region.width,
            "height": avg_region.height,
            "avg_distortion": avg_region.avg_distortion,
        }
        fragment_info.setdefault("variance", {})[res_key] = {
            "x": var_region.x,
            "y": var_region.y,
            "width": var_region.width,
            "height": var_region.height,
            "avg_distortion": var_region.avg_distortion,
        }

    return all_records, fragment_info


def _encode_and_measure(
    source_image: str,
    original_image: str,
    fmt: str,
    quality: int,
    project_root_str: str,
    chroma_subsampling: str | None = None,
    speed: int | None = None,
    effort: int | None = None,
    method: int | None = None,
    resolution: int | None = None,
    extra_args: dict[str, str | int | bool] | None = None,
    save_dir_str: str | None = None,
    source_image_label: str | None = None,
) -> QualityRecord:
    """Encode a single image variant and measure its quality.

    This is a **top-level** function (no closures) so that it can be
    submitted to a :class:`~concurrent.futures.ProcessPoolExecutor` with
    the ``"spawn"`` multiprocessing context.

    Encoded files are written to a temporary directory and
    automatically cleaned up after measurement.

    Args:
        source_image: Absolute path to the image to encode (may be
            preprocessed).
        original_image: Absolute path to the original dataset image.
        fmt: Encoding format (``"jpeg"``, ``"webp"``, ``"avif"``, ``"jxl"``).
        quality: Quality setting (0–100).
        project_root_str: Absolute path of the project root (as string,
            for pickling).
        chroma_subsampling: Optional chroma subsampling mode.
        speed: Optional AVIF speed setting.
        effort: Optional JXL effort setting.
        method: Optional WebP method setting.
        resolution: Resolution tag (if preprocessed).
        extra_args: Extra encoder arguments.
        save_dir_str: If given, copy the encoded file to this directory.
        source_image_label: Relative path string to use for ``source_image``
            in the output record (avoids leaking temp-dir paths).

    Returns:
        A :class:`QualityRecord` with encoding + quality data.
    """
    from PIL import Image

    source_path = Path(source_image)
    original_path = Path(original_image)
    project_root = Path(project_root_str)

    # Labels for the output record
    src_label = source_image_label or _make_rel(source_path, project_root)
    orig_label = _make_rel(original_path, project_root)

    # Read source dimensions / size upfront so error records can include them
    try:
        with Image.open(source_path) as img:
            width, height = img.size
        source_file_size = source_path.stat().st_size
    except Exception:
        width, height, source_file_size = 0, 0, 0

    output_name = _build_output_name(
        source_path.stem,
        quality,
        chroma_subsampling=chroma_subsampling,
        speed=speed,
        effort=effort,
        method=method,
        resolution=resolution,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = ImageEncoder(Path(tmpdir))

        # --- Encode -----------------------------------------------------------
        t0 = time.monotonic()
        try:
            if fmt == "jpeg":
                result = encoder.encode_jpeg(source_path, quality, output_name=output_name)
            elif fmt == "webp":
                m = method if method is not None else 4
                result = encoder.encode_webp(
                    source_path, quality, method=m, output_name=output_name
                )
            elif fmt == "avif":
                s = speed if speed is not None else 6
                result = encoder.encode_avif(
                    source_path,
                    quality,
                    speed=s,
                    chroma_subsampling=chroma_subsampling,
                    output_name=output_name,
                )
            elif fmt == "jxl":
                e = effort if effort is not None else 7
                result = encoder.encode_jxl(source_path, quality, effort=e, output_name=output_name)
            else:
                return (
                    _error_record(
                        src_label,
                        orig_label,
                        fmt,
                        quality,
                        width,
                        height,
                        source_file_size,
                        chroma_subsampling,
                        speed,
                        effort,
                        method,
                        resolution,
                        extra_args,
                        f"Unknown format: {fmt}",
                    ),
                    None,
                )
        except Exception as exc:
            return (
                _error_record(
                    src_label,
                    orig_label,
                    fmt,
                    quality,
                    width,
                    height,
                    source_file_size,
                    chroma_subsampling,
                    speed,
                    effort,
                    method,
                    resolution,
                    extra_args,
                    f"Encoding exception: {exc}",
                ),
                None,
            )

        encoding_time = time.monotonic() - t0

        if not result.success or result.output_path is None:
            return (
                _error_record(
                    src_label,
                    orig_label,
                    fmt,
                    quality,
                    width,
                    height,
                    source_file_size,
                    chroma_subsampling,
                    speed,
                    effort,
                    method,
                    resolution,
                    extra_args,
                    f"Encoding failed: {result.error_message}",
                ),
                None,
            )

        file_size = result.file_size or result.output_path.stat().st_size

        # --- Measure ----------------------------------------------------------
        # Use butteraugli with --rawdistmap to get both the aggregate
        # score and the per-pixel distortion map in a single invocation.
        distmap_pfm_path = Path(tmpdir) / f"{output_name}.pfm"
        try:
            measurer = QualityMeasurer()
            metrics = measurer.measure_all(
                source_path, result.output_path, distmap_path=distmap_pfm_path,
            )
        except Exception as exc:
            return (
                _error_record(
                    src_label,
                    orig_label,
                    fmt,
                    quality,
                    width,
                    height,
                    source_file_size,
                    chroma_subsampling,
                    speed,
                    effort,
                    method,
                    resolution,
                    extra_args,
                    f"Measurement exception: {exc}",
                    file_size=file_size,
                    encoding_time=encoding_time,
                ),
                None,
            )

        # Read distortion map into numpy (stays in-process, no pickling)
        distmap_arr: np.ndarray | None = None
        if distmap_pfm_path.exists():
            try:
                distmap_arr = read_pfm(distmap_pfm_path)
            except (ValueError, OSError):
                pass  # Non-fatal; fragment detection will be skipped

        # --- Optionally persist artifact --------------------------------------
        encoded_path_label = ""
        if save_dir_str is not None:
            save_dir = Path(save_dir_str)
            save_dir.mkdir(parents=True, exist_ok=True)
            dest = save_dir / result.output_path.name
            shutil.copy2(result.output_path, dest)
            encoded_path_label = _make_rel(dest, project_root)
            # Also persist the distortion map PFM alongside the encoded file
            if distmap_pfm_path.exists():
                pfm_dest = save_dir / distmap_pfm_path.name
                shutil.copy2(distmap_pfm_path, pfm_dest)

    # tmpdir auto-cleaned here
    return (
        QualityRecord(
            source_image=src_label,
            original_image=orig_label,
            encoded_path=encoded_path_label,
            format=fmt,
            quality=quality,
            file_size=file_size,
            width=width,
            height=height,
            source_file_size=source_file_size,
            ssimulacra2=metrics.ssimulacra2,
            psnr=metrics.psnr,
            ssim=metrics.ssim,
            butteraugli=metrics.butteraugli,
            encoding_time=encoding_time,
            chroma_subsampling=chroma_subsampling,
            speed=speed,
            effort=effort,
            method=method,
            resolution=resolution,
            extra_args=extra_args,
            measurement_error=metrics.error_message,
        ),
        distmap_arr,
    )


def _error_record(
    source_image: str,
    original_image: str,
    fmt: str,
    quality: int,
    width: int,
    height: int,
    source_file_size: int,
    chroma_subsampling: str | None,
    speed: int | None,
    effort: int | None,
    method: int | None,
    resolution: int | None,
    extra_args: dict[str, str | int | bool] | None,
    error: str,
    *,
    file_size: int = 0,
    encoding_time: float | None = None,
) -> QualityRecord:
    """Build a :class:`QualityRecord` that records an error."""
    return QualityRecord(
        source_image=source_image,
        original_image=original_image,
        encoded_path="",
        format=fmt,
        quality=quality,
        file_size=file_size,
        width=width,
        height=height,
        source_file_size=source_file_size,
        ssimulacra2=None,
        psnr=None,
        ssim=None,
        butteraugli=None,
        encoding_time=encoding_time,
        chroma_subsampling=chroma_subsampling,
        speed=speed,
        effort=effort,
        method=method,
        resolution=resolution,
        extra_args=extra_args,
        measurement_error=error,
    )


# ---------------------------------------------------------------------------
# Worst-image detection
# ---------------------------------------------------------------------------


def _find_worst_original(
    records: list[QualityRecord],
    strategy: str = "average",
) -> str | None:
    """Return the ``original_image`` path with the worst SSIMULACRA2.

    Supports two strategies:

    * ``"average"`` — groups records by ``original_image`` and returns
      the one whose mean SSIMULACRA2 is lowest.
    * ``"variance"`` — groups records by ``original_image`` and returns
      the one whose SSIMULACRA2 variance is highest.

    Returns ``None`` if no valid scores exist.
    """
    image_scores: dict[str, list[float]] = {}
    for rec in records:
        if rec.measurement_error is None and rec.ssimulacra2 is not None:
            key = rec.original_image
            if key not in image_scores:
                image_scores[key] = []
            image_scores[key].append(rec.ssimulacra2)

    if not image_scores:
        return None

    if strategy == "variance":
        image_var: dict[str, float] = {}
        for img, scores in image_scores.items():
            if len(scores) < 2:
                image_var[img] = 0.0
            else:
                mean = sum(scores) / len(scores)
                image_var[img] = sum((s - mean) ** 2 for s in scores) / len(scores)
        return max(image_var, key=lambda k: image_var[k])

    # strategy == "average"
    image_avgs = {img: sum(s) / len(s) for img, s in image_scores.items()}
    return min(image_avgs, key=lambda k: image_avgs[k])


# ---------------------------------------------------------------------------
# Task generation helpers
# ---------------------------------------------------------------------------


def _expand_encoder_tasks(
    source_image: Path,
    original_image: Path,
    enc: EncoderConfig,
    save_artifact_dir: Path | None,
    study_id: str,
) -> list[dict[str, Any]]:
    """Expand one :class:`EncoderConfig` into keyword-arg dicts for
    :func:`_encode_and_measure`.

    Resolution is now part of the Cartesian product alongside quality,
    chroma, speed, effort, and method.

    Returns a list of dicts that can be unpacked as
    ``_encode_and_measure(**kw)`` (all values are pickle-safe).
    """
    chroma_options: list[str | None] = (
        list(enc.chroma_subsampling) if enc.chroma_subsampling else [None]
    )
    speed_options: list[int | None] = list(enc.speed) if enc.speed else [None]
    effort_options: list[int | None] = list(enc.effort) if enc.effort else [None]
    method_options: list[int | None] = list(enc.method) if enc.method else [None]
    resolution_options: list[int | None] = list(enc.resolution) if enc.resolution else [None]

    tasks: list[dict[str, Any]] = []
    for resolution in resolution_options:
        res_label = f"r{resolution}" if resolution else "original"

        if save_artifact_dir is not None:
            save_dir_str: str | None = str(save_artifact_dir / enc.format / res_label)
        else:
            save_dir_str = None

        # Build source_image_label for preprocessed images
        if resolution is not None:
            source_image_label: str | None = (
                f"data/preprocessed/{study_id}"
                f"/r{resolution}/{source_image.stem}_r{resolution}.png"
            )
        else:
            source_image_label = None

        for q in enc.quality:
            for chroma in chroma_options:
                for spd in speed_options:
                    for eff in effort_options:
                        for mth in method_options:
                            tasks.append(
                                {
                                    "source_image": str(source_image),
                                    "original_image": str(original_image),
                                    "fmt": enc.format,
                                    "quality": q,
                                    "chroma_subsampling": chroma,
                                    "speed": spd,
                                    "effort": eff,
                                    "method": mth,
                                    "resolution": resolution,
                                    "extra_args": enc.extra_args,
                                    "save_dir_str": save_dir_str,
                                    "source_image_label": source_image_label,
                                }
                            )
    return tasks


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


class PipelineRunner:
    """Merged encode+measure pipeline with time-budget support.

    Uses a worker-per-image architecture where each worker processes one
    complete image before moving to the next. For each image, the worker:

    1. Preprocesses (resize) for every configured resolution.
    2. Encodes all parameter combinations sequentially.
    3. Measures quality of each encoded variant.
    4. Pulls the next image from the queue if time budget allows.

    This architecture keeps all workers fully utilized and naturally
    staggers memory-intensive operations across workers, reducing peak
    memory usage.

    Time budget behavior:
    - Initial batch fills all available workers (max throughput at start)
    - Budget is checked before submitting additional images
    - When budget expires, new submissions stop but in-flight work completes
    - Note: In-flight images process sequentially on their assigned workers,
      which may leave some workers idle during the finish phase. A future
      optimization could switch to task-level parallelism after budget expiry.

    Encoded files live in a temporary directory and are discarded
    after measurement unless ``save_artifacts=True``.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.data_dir = project_root / "data"

        # Worst-image tracking state per strategy (used by _update_worst_image).
        # Fragment-level comparison: higher distortion score = worse.
        # Each strategy tracks: staging_dir, score, original_key, fragment_info.
        self._worst: dict[str, dict] = {
            "average": {
                "staging_dir": None,
                "score": float("-inf"),    # higher avg distortion is worse
                "original_key": None,
                "fragment_info": None,
            },
            "variance": {
                "staging_dir": None,
                "score": float("-inf"),    # higher var distortion is worse
                "original_key": None,
                "fragment_info": None,
            },
        }

    # ------------------------------------------------------------------
    # Image collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_images(dataset_dir: Path, max_images: int | None = None) -> list[Path]:
        """Return sorted image paths from *dataset_dir*."""
        extensions = {".png", ".jpg", ".jpeg", ".ppm", ".pgm", ".bmp", ".tiff", ".tif"}
        images = sorted(
            p for p in dataset_dir.rglob("*") if p.suffix.lower() in extensions and p.is_file()
        )
        if max_images is not None:
            images = images[:max_images]
        return images

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_image(
        image_path: Path,
        resolution: int,
        output_dir: Path,
    ) -> Path:
        """Resize a single image and return the path to the resized copy."""
        preprocessor = ImagePreprocessor(output_dir)
        output_name = f"{image_path.stem}_r{resolution}.png"
        return preprocessor.resize_image(
            image_path,
            target_size=(resolution, resolution),
            output_name=output_name,
            keep_aspect_ratio=True,
        )

    # ------------------------------------------------------------------
    # Tool versions
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_tool_versions() -> dict[str, str]:
        versions: dict[str, str] = {}
        for tool in ("ssimulacra2", "butteraugli", "ffmpeg"):
            v = get_measurement_tool_version(tool)
            if v:
                versions[tool] = v
        for encoder in ("cjpeg", "cwebp", "avifenc", "cjxl"):
            v = get_encoder_version(encoder)
            if v:
                versions[encoder] = v
        return versions

    # ------------------------------------------------------------------
    # Worst-image tracking helpers
    # ------------------------------------------------------------------

    def _update_worst_image(
        self,
        image_records: list[QualityRecord],
        img_path: Path,
        staging_base: Path,
        fragment_info: dict[str, dict],
    ) -> None:
        """Compare a newly-completed image against the current worst for each strategy.

        Uses **fragment-level** distortion scores from the Butteraugli
        distortion maps rather than image-level SSIMULACRA2.  For both
        ``"average"`` and ``"variance"`` strategies, the image with the
        highest worst-fragment distortion score is considered worst.

        When the study uses multiple resolutions, the per-image score is
        the maximum fragment score across all resolutions.

        Staging directories that are no longer the worst for any strategy
        are deleted immediately to save disk space.

        Internal state is stored in ``self._worst`` dict keyed by
        strategy name.
        """
        candidate_dir = staging_base / img_path.stem
        original_key = image_records[0].original_image if image_records else None

        # Compute per-strategy fragment score (max across resolutions)
        candidate_scores: dict[str, float] = {}
        for strat in ("average", "variance"):
            regions = fragment_info.get(strat, {})
            if regions:
                candidate_scores[strat] = max(
                    r["avg_distortion"] for r in regions.values()
                )

        if not candidate_scores:
            # No fragments (butteraugli unavailable or all measurements failed)
            shutil.rmtree(candidate_dir, ignore_errors=True)
            return

        # Determine if candidate is worst for each strategy
        is_worst_avg = candidate_scores.get("average", float("-inf")) > self._worst["average"]["score"]
        is_worst_var = candidate_scores.get("variance", float("-inf")) > self._worst["variance"]["score"]

        # Collect dirs being replaced
        dirs_released: set[Path] = set()

        if is_worst_avg:
            old_dir = self._worst["average"]["staging_dir"]
            if old_dir is not None:
                dirs_released.add(old_dir)
            self._worst["average"] = {
                "staging_dir": candidate_dir,
                "score": candidate_scores.get("average", float("-inf")),
                "original_key": original_key,
                "fragment_info": fragment_info.get("average"),
            }

        if is_worst_var:
            old_dir = self._worst["variance"]["staging_dir"]
            if old_dir is not None:
                dirs_released.add(old_dir)
            self._worst["variance"] = {
                "staging_dir": candidate_dir,
                "score": candidate_scores.get("variance", float("-inf")),
                "original_key": original_key,
                "fragment_info": fragment_info.get("variance"),
            }

        # A dir is still needed if it's the current staging_dir for either strategy
        active_dirs = {
            self._worst["average"]["staging_dir"],
            self._worst["variance"]["staging_dir"],
        }
        for d in dirs_released:
            if d not in active_dirs:
                shutil.rmtree(d, ignore_errors=True)

        # If candidate was not claimed by either strategy, delete it
        if not is_worst_avg and not is_worst_var and candidate_dir not in active_dirs:
            shutil.rmtree(candidate_dir, ignore_errors=True)

    def _finalize_worst_image(
        self,
        study_id: str,
        staging_base: Path,
        all_records: list[QualityRecord],
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Move worst image artifacts to final locations and return metadata.

        Handles both ``"average"`` and ``"variance"`` strategies.  Each
        strategy's files are placed in a dedicated subdirectory under
        ``data/encoded/<study_id>/`` (e.g. ``average/`` and ``variance/``).
        When both strategies select the same image, the files are moved
        once and copied for the second strategy.

        Returns:
            ``(worst_images_meta, worst_fragments_meta)`` — two dicts
            keyed by strategy.

            ``worst_images_meta`` contains ``original_image`` and
            ``score`` for each worst image.

            ``worst_fragments_meta`` contains per-resolution fragment
            positions (``{strategy: {resolution: {x, y, width, height,
            avg_distortion}}}``).
        """
        staging_base_prefix = _make_rel(staging_base, self.project_root)
        final_base = self.data_dir / "encoded" / study_id
        worst_images_meta: dict[str, dict] = {}
        worst_fragments_meta: dict[str, dict] = {}

        # Group strategies by their staging dir so we handle shared images once
        staging_to_strats: dict[Path, list[str]] = {}
        for strat in ("average", "variance"):
            info = self._worst[strat]
            sd: Path | None = info["staging_dir"]
            if sd is not None and sd.exists():
                staging_to_strats.setdefault(sd, []).append(strat)

        for staging_dir, strategies in staging_to_strats.items():
            staging_prefix = _make_rel(staging_dir, self.project_root)
            first_strat = strategies[0]
            first_final = final_base / first_strat
            first_final_prefix = _make_rel(first_final, self.project_root)

            # Move files to first strategy's final dir
            for src_file in staging_dir.rglob("*"):
                if src_file.is_file():
                    rel = src_file.relative_to(staging_dir)
                    dest = first_final / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_file), str(dest))

            # Copy to additional strategy dirs if both selected the same image
            for extra_strat in strategies[1:]:
                extra_final = final_base / extra_strat
                shutil.copytree(first_final, extra_final, dirs_exist_ok=True)

            # Update encoded_path in records that came from this staging dir
            for rec in all_records:
                if rec.encoded_path and rec.encoded_path.startswith(staging_prefix):
                    rec.encoded_path = rec.encoded_path.replace(
                        staging_prefix, first_final_prefix, 1,
                    )

            # Build metadata for each strategy
            for strat in strategies:
                info = self._worst[strat]
                worst_images_meta[strat] = {
                    "original_image": info["original_key"],
                    "score": info["score"],
                }

                # Fragment metadata
                frag = info.get("fragment_info")
                if frag is not None:
                    # Convert None keys to "null" for JSON serialisation
                    worst_fragments_meta[strat] = {
                        "original_image": info["original_key"],
                        "score": info["score"],
                        "regions": {
                            (str(k) if k is not None else "null"): v
                            for k, v in frag.items()
                        },
                    }

                score_label = (
                    f"avg distortion score: {info['score']:.4f}"
                    if strat == "average"
                    else f"distortion variance score: {info['score']:.4f}"
                )
                print(f"\n  [{strat}] Worst image artifacts saved to: "
                      f"{final_base / strat}")
                if info["original_key"]:
                    print(f"  [{strat}] Image: {info['original_key']} "
                          f"({score_label})")

        # Clear remaining staging paths from records (non-worst images)
        for rec in all_records:
            if rec.encoded_path and rec.encoded_path.startswith(staging_base_prefix):
                rec.encoded_path = ""

        # Cleanup staging base
        shutil.rmtree(staging_base, ignore_errors=True)

        return worst_images_meta, worst_fragments_meta

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        config: StudyConfig,
        *,
        time_budget: float | None = None,
        save_artifacts: bool = False,
        save_worst_image: bool = False,
        num_workers: int | None = None,
    ) -> QualityResults:
        """Run the merged encode+measure pipeline.

        Args:
            config: Study configuration describing dataset, encoders,
                and optional preprocessing.
            time_budget: Maximum wall-clock seconds to spend.  When set,
                the pipeline processes images until this budget is
                exhausted (always completing the current image).
                ``None`` means process all available images.
            save_artifacts: If ``True``, persist encoded files to
                ``data/encoded/<study_id>/``.
            save_worst_image: If ``True``, re-encode the worst-quality
                source image (by average SSIMULACRA2) and persist its
                encoded files to ``data/encoded/<study_id>/``.  This is
                cheaper than ``save_artifacts`` (only one image) and
                provides the files needed by the visual-comparison tool.
            num_workers: Parallel workers (default: CPU count).

        Returns:
            :class:`QualityResults` ready for analysis / report.

        Raises:
            ValueError: If dataset is not found in configuration.
            FileNotFoundError: If dataset is not downloaded or has no images.
        """
        # --- Resolve dataset --------------------------------------------------
        config_file = self.project_root / "config" / "datasets.json"
        fetcher = DatasetFetcher(
            self.data_dir / "datasets",
            config_file=config_file if config_file.exists() else None,
        )
        ds_config = fetcher.get_dataset_config(config.dataset_id)
        if ds_config is None:
            msg = f"Dataset '{config.dataset_id}' not found in configuration"
            raise ValueError(msg)

        dataset_name = ds_config.rename_to or config.dataset_id
        dataset_dir = self.data_dir / "datasets" / dataset_name
        if not dataset_dir.exists():
            msg = (
                f"Dataset directory not found: {dataset_dir}. "
                f"Fetch it first with: just fetch {config.dataset_id}"
            )
            raise FileNotFoundError(msg)

        all_images = self._collect_images(dataset_dir, config.max_images)
        if not all_images:
            msg = f"No images found in {dataset_dir}"
            raise FileNotFoundError(msg)

        # --- Resolutions (for banner display only) ----------------------------
        all_resolutions: set[int] = set()
        for enc in config.encoders:
            if enc.resolution:
                all_resolutions.update(enc.resolution)

        # --- Workers ----------------------------------------------------------
        if num_workers is None:
            num_workers = os.cpu_count() or 1

        save_artifact_dir = (self.data_dir / "encoded" / config.id) if save_artifacts else None

        # --- Banner -----------------------------------------------------------
        print(f"Pipeline: {config.name}")
        print(f"Dataset: {config.dataset_id} ({len(all_images)} images available)")
        if time_budget is not None:
            print(f"Time budget: {_format_duration(time_budget)}")
        else:
            print("Time budget: unlimited (processing all images)")
        if all_resolutions:
            res_labels = sorted(f"r{r}" for r in all_resolutions)
            print(f"Resolutions: {', '.join(res_labels)}")
        else:
            print("Resolutions: original")
        print(f"Workers: {num_workers}")
        if save_artifacts:
            print(f"Saving artifacts to: {save_artifact_dir}")
        if save_worst_image and not save_artifacts:
            print("Saving worst image artifacts (streaming)")
        print()

        # --- Prepare config dict for serialization ----------------------------
        # Convert StudyConfig to a dict for pickling (EncoderConfig → dict)
        config_dict = {
            "id": config.id,
            "name": config.name,
            "encoders": [
                {
                    "format": enc.format,
                    "quality": enc.quality,
                    "chroma_subsampling": enc.chroma_subsampling,
                    "speed": enc.speed,
                    "effort": enc.effort,
                    "method": enc.method,
                    "resolution": enc.resolution,
                    "extra_args": enc.extra_args,
                }
                for enc in config.encoders
            ],
        }

        # --- Process images with worker-per-image model -----------------------
        # Each worker processes one complete image (all tasks sequentially),
        # then pulls the next image from the queue. This keeps workers fully
        # utilized and staggers memory-intensive operations.
        all_records: list[QualityRecord] = []
        start_time = time.monotonic()
        images_processed = 0

        # --- Worst-image tracking state (streaming) --------------------------
        # When save_worst_image is True, each worker saves its encoded files
        # to a per-image staging directory.  After each worker completes, the
        # main process compares its scores with the current worst for both
        # "average" and "variance" strategies and immediately deletes the
        # staging directory of the loser.
        track_worst = save_worst_image and not save_artifacts
        staging_base: Path | None = None

        # Reset worst-image tracking state
        self._worst = {
            "average": {
                "staging_dir": None, "score": float("-inf"),
                "original_key": None, "fragment_info": None,
            },
            "variance": {
                "staging_dir": None, "score": float("-inf"),
                "original_key": None, "fragment_info": None,
            },
        }

        if track_worst:
            staging_base = self.data_dir / "encoded" / f".staging-{config.id}"
            if staging_base.exists():
                shutil.rmtree(staging_base)
            staging_base.mkdir(parents=True)

        mp_ctx = multiprocessing.get_context("spawn")

        # Helper to build per-image staging dir path
        def _img_staging(img_path: Path) -> str | None:
            if staging_base is None:
                return None
            d = staging_base / img_path.stem
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            # Submit initial batch of images (fill all workers)
            # This ensures maximum throughput and no idle workers at startup.
            # Time budget is checked when submitting additional images beyond this batch.
            pending_images = all_images.copy()

            # Track futures in a set (not dict) so we can add during iteration
            pending_futures: set[Any] = set()
            future_to_image: dict[Any, Path] = {}

            initial_count = min(num_workers, len(pending_images))

            for _ in range(initial_count):
                img_path = pending_images.pop(0)
                future = executor.submit(
                    _process_image,
                    image_path_str=str(img_path),
                    config_dict=config_dict,
                    project_root_str=str(self.project_root),
                    save_artifacts=save_artifacts,
                    save_artifact_dir_str=_img_staging(img_path),
                )
                pending_futures.add(future)
                future_to_image[future] = img_path

            # Process completed images and submit new ones as workers become available
            budget_exceeded = False
            while pending_futures:
                # Wait for next completion
                done, pending_futures = wait(pending_futures, return_when="FIRST_COMPLETED")

                for future in done:
                    img_path = future_to_image[future]
                    image_records, fragment_info = future.result()

                    all_records.extend(image_records)
                    images_processed += 1

                    # --- Worst-image streaming update -------------------------
                    if track_worst and staging_base is not None:
                        self._update_worst_image(
                            image_records,
                            img_path,
                            staging_base,
                            fragment_info,
                        )

                    # Progress
                    elapsed = time.monotonic() - start_time
                    total_tasks = len(image_records)
                    errors = sum(1 for r in image_records if r.measurement_error is not None)
                    err_str = f" | {errors} errors" if errors else ""
                    budget_str = ""
                    if time_budget is not None:
                        remaining = max(0.0, time_budget - elapsed)
                        budget_str = f" | {_format_duration(remaining)} remaining"

                    # Show if this is post-budget work being completed
                    status_prefix = "  "
                    if budget_exceeded:
                        status_prefix = "  (finishing) "

                    print(
                        f"{status_prefix}[{images_processed}/{len(all_images)}] "
                        f"{img_path.name}: {total_tasks} tasks{err_str} "
                        f"[{_format_duration(elapsed)} elapsed{budget_str}]"
                    )

                    # Check time budget and submit next image if available
                    # Budget is only checked for images beyond the initial worker-filling batch
                    # Once budget is exceeded, stop submitting but continue collecting in-flight work
                    if pending_images and not budget_exceeded:
                        if time_budget is None or (time.monotonic() - start_time) < time_budget:
                            next_img = pending_images.pop(0)
                            next_future = executor.submit(
                                _process_image,
                                image_path_str=str(next_img),
                                config_dict=config_dict,
                                project_root_str=str(self.project_root),
                                save_artifacts=save_artifacts,
                                save_artifact_dir_str=_img_staging(next_img),
                            )
                            pending_futures.add(next_future)
                            future_to_image[next_future] = next_img
                        elif not budget_exceeded:
                            # Budget exhausted, stop submitting new work
                            # but continue collecting results from in-flight images
                            print(
                                f"\nTime budget reached "
                                f"({_format_duration(time.monotonic() - start_time)} >= "
                                f"{_format_duration(time_budget)})"
                            )
                            print(
                                f"  Waiting for {len(pending_futures)} in-flight images to complete..."
                            )
                            budget_exceeded = True

        elapsed_total = time.monotonic() - start_time

        # --- Finalize worst-image artifacts -----------------------------------
        worst_images_meta: dict[str, dict] | None = None
        worst_fragments_meta: dict[str, dict] | None = None
        if track_worst and staging_base is not None:
            worst_images_meta, worst_fragments_meta = self._finalize_worst_image(
                config.id, staging_base, all_records,
            )

        # --- Compute worst-image metadata for JSON output ---------------------
        if worst_images_meta is None:
            worst_images_meta = {}
            for strat in ("average", "variance"):
                worst_key = _find_worst_original(all_records, strategy=strat)
                if worst_key is not None:
                    # Compute the score
                    strat_scores: dict[str, list[float]] = {}
                    for rec in all_records:
                        if rec.measurement_error is None and rec.ssimulacra2 is not None:
                            key = rec.original_image
                            if key not in strat_scores:
                                strat_scores[key] = []
                            strat_scores[key].append(rec.ssimulacra2)
                    scores = strat_scores.get(worst_key, [])
                    if scores:
                        mean = sum(scores) / len(scores)
                        if strat == "average":
                            score = mean
                        elif len(scores) >= 2:
                            score = sum((s - mean) ** 2 for s in scores) / len(scores)
                        else:
                            score = 0.0
                        worst_images_meta[strat] = {
                            "original_image": worst_key,
                            "score": score,
                        }

        # --- Assemble results -------------------------------------------------
        tool_versions = self._collect_tool_versions()

        results = QualityResults(
            study_id=config.id,
            study_name=config.name,
            dataset={
                "id": config.dataset_id,
                "path": _make_rel(dataset_dir, self.project_root),
                "image_count": images_processed,
            },
            measurements=all_records,
            timestamp=datetime.now(UTC).isoformat(),
            tool_versions=tool_versions,
            worst_images=worst_images_meta if worst_images_meta else None,
            worst_fragments=worst_fragments_meta if worst_fragments_meta else None,
        )

        # --- Summary ----------------------------------------------------------
        total_errors = sum(1 for r in all_records if r.measurement_error is not None)
        print(f"\nPipeline complete in {_format_duration(elapsed_total)}")
        print(f"  Images processed: {images_processed}/{len(all_images)}")
        print(f"  Total measurements: {len(all_records)}")
        if total_errors:
            print(f"  Errors: {total_errors}")

        return results
