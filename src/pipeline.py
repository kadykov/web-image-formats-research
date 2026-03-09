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
from src.quality import (
    QualityMeasurer,
    QualityRecord,
    QualityResults,
    get_measurement_tool_version,
)
from src.study import EncoderConfig, StudyConfig

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
    crop: int | None = None,
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
    if crop is not None:
        parts.append(f"c{crop}")
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

    Resolution and crop are part of the per-encoder Cartesian product.
    Preprocessing (resizing / cropping) is handled inline with
    in-worker caches so each level is only computed once per image.

    For crop-impact studies the worker first performs a **fragment
    selection phase**: the image is encoded at the lowest quality with
    the first encoder, a Butteraugli distortion map is measured, and
    the most distorted analysis fragment is identified.  All subsequent
    tasks then use this shared fragment for quality measurement.

    Args:
        image_path_str: Absolute path to the source image.
        config_dict: Serializable dict with study config data.
        project_root_str: Project root path (for pickling).
        save_artifacts: Whether to save encoded files.
        save_artifact_dir_str: Override the artifact directory. When
            set, encoded files are saved here instead of the default
            ``data/encoded/<study_id>/`` path.

    Returns:
        A list of :class:`QualityRecord` objects for all tasks.
    """
    import tempfile

    from src.preprocessing import CropResult, ImagePreprocessor
    from src.quality import (
        QualityMeasurer as _QM,
        extract_fragment as _extract_frag,
        find_worst_region_in_array,
        read_pfm,
    )
    from src.study import EncoderConfig

    image_path = Path(image_path_str)
    project_root = Path(project_root_str)
    study_id = config_dict["id"]
    analysis_fragment_size: int = config_dict.get("analysis_fragment_size") or 200
    crop_too_small_strategy: str = config_dict.get(
        "crop_too_small_strategy", "skip_image"
    )

    # Reconstruct encoder configs
    encoders = [EncoderConfig(**enc_dict) for enc_dict in config_dict["encoders"]]

    # Determine artifact save directory
    if save_artifact_dir_str is not None:
        save_artifact_dir: Path | None = Path(save_artifact_dir_str)
    elif save_artifacts:
        save_artifact_dir = project_root / "data" / "encoded" / study_id
    else:
        save_artifact_dir = None

    # Check if any encoder uses crop
    uses_crop = any(enc.crop for enc in encoders)

    all_records: list[QualityRecord] = []

    # In-worker preprocessing caches
    preprocessed_cache: dict[int, Path] = {}  # resolution → resized path
    crop_cache: dict[int, CropResult] = {}    # crop_level → CropResult

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

        def _get_cropped_source(
            crop_level: int,
            fragment: dict[str, int],
        ) -> CropResult:
            """Get cropped source image (cached)."""
            if crop_level in crop_cache:
                return crop_cache[crop_level]
            crop_dir = Path(prep_tmpdir) / f"c{crop_level}"
            preprocessor = ImagePreprocessor(crop_dir)
            output_name = f"{image_path.stem}_c{crop_level}.png"
            result = preprocessor.crop_image_around_fragment(
                image_path,
                fragment=fragment,
                target_longest_edge=crop_level,
                output_name=output_name,
                adjust_aspect_ratio=(
                    crop_too_small_strategy == "adjust_aspect_ratio"
                ),
            )
            crop_cache[crop_level] = result
            return result

        # ------------------------------------------------------------------
        # Fragment selection phase (crop-impact studies only)
        # ------------------------------------------------------------------
        # Encode at the lowest quality of the first encoder using the
        # full-resolution image and generate a Butteraugli distortion
        # map.  The worst region in this map becomes the shared analysis
        # fragment for all tasks on this image.
        analysis_fragment: dict[str, int] | None = None

        if uses_crop:
            first_crop_enc = next(e for e in encoders if e.crop)
            lowest_q = min(first_crop_enc.quality)
            fmt0 = first_crop_enc.format

            frag_dir = Path(prep_tmpdir) / "fragment_selection"
            frag_dir.mkdir(exist_ok=True)

            from src.encoder import ImageEncoder as _IE

            _enc0 = _IE(frag_dir)
            _speed0 = (min(first_crop_enc.speed) if first_crop_enc.speed else None)
            _effort0 = (min(first_crop_enc.effort) if first_crop_enc.effort else None)
            _method0 = (min(first_crop_enc.method) if first_crop_enc.method else None)

            try:
                if fmt0 == "jpeg":
                    _r0 = _enc0.encode_jpeg(image_path, lowest_q)
                elif fmt0 == "webp":
                    _r0 = _enc0.encode_webp(image_path, lowest_q, method=_method0 or 4)
                elif fmt0 == "avif":
                    _r0 = _enc0.encode_avif(image_path, lowest_q, speed=_speed0 or 6)
                elif fmt0 == "jxl":
                    _r0 = _enc0.encode_jxl(image_path, lowest_q, effort=_effort0 or 7)
                else:
                    _r0 = None  # type: ignore[assignment]

                if _r0 is not None and _r0.success and _r0.output_path is not None:
                    pfm_path = frag_dir / "distmap.pfm"
                    _meas = _QM()
                    _meas.measure_butteraugli_with_distmap(
                        image_path, _r0.output_path, pfm_path
                    )
                    if pfm_path.exists():
                        dm = read_pfm(pfm_path)
                        region = find_worst_region_in_array(
                            dm, crop_size=analysis_fragment_size
                        )
                        analysis_fragment = {
                            "x": region.x,
                            "y": region.y,
                            "width": region.width,
                            "height": region.height,
                        }
            except Exception:
                # If fragment selection fails, fall back to top-left corner
                pass

            if analysis_fragment is None:
                # Fallback: use top-left corner
                from PIL import Image as _PILImage
                with _PILImage.open(image_path) as _im:
                    _iw, _ih = _im.size
                analysis_fragment = {
                    "x": 0,
                    "y": 0,
                    "width": min(analysis_fragment_size, _iw),
                    "height": min(analysis_fragment_size, _ih),
                }

        # ------------------------------------------------------------------
        # skip_image pre-check: verify all crop levels fit the fragment
        # ------------------------------------------------------------------
        if uses_crop and analysis_fragment is not None:
            if crop_too_small_strategy == "skip_image":
                from PIL import Image as _PILImg

                with _PILImg.open(image_path) as _chk:
                    _cw, _ch = _chk.size
                _longest = max(_cw, _ch)
                fw = analysis_fragment["width"]
                fh = analysis_fragment["height"]
                all_crop_levels: set[int] = set()
                for enc in encoders:
                    if enc.crop:
                        all_crop_levels.update(enc.crop)
                for cl in all_crop_levels:
                    if cl >= _longest:
                        continue
                    sc = cl / _longest
                    cw_check = max(1, round(_cw * sc))
                    ch_check = max(1, round(_ch * sc))
                    if fw > cw_check or fh > ch_check:
                        import warnings

                        warnings.warn(
                            f"Skipping {image_path.name}: crop level "
                            f"{cl} produces {cw_check}x{ch_check} "
                            f"which cannot fit analysis fragment "
                            f"{fw}x{fh}",
                            stacklevel=1,
                        )
                        return all_records  # empty

        # ------------------------------------------------------------------
        # Main encoding loop
        # ------------------------------------------------------------------
        for enc in encoders:
            tasks = _expand_encoder_tasks(
                source_image=image_path,
                original_image=image_path,
                enc=enc,
                save_artifact_dir=save_artifact_dir,
                study_id=study_id,
                analysis_fragment=analysis_fragment if enc.crop else None,
            )

            # Execute tasks sequentially within this worker
            for task_kw in tasks:
                # Resolve the actual source image for this task
                resolution = task_kw["resolution"]
                crop_level = task_kw.get("crop")

                if crop_level is not None and analysis_fragment is not None:
                    # Crop-impact mode: crop the original image around the fragment
                    try:
                        crop_result = _get_cropped_source(
                            crop_level, analysis_fragment
                        )
                    except ValueError:
                        if crop_too_small_strategy == "skip_measurement":
                            import warnings

                            warnings.warn(
                                f"Skipping {image_path.name} at crop "
                                f"{crop_level}: fragment does not fit",
                                stacklevel=1,
                            )
                            continue
                        raise
                    task_kw["source_image"] = str(crop_result.path)
                    task_kw["crop_region"] = crop_result.crop_region

                    # Transform fragment coordinates from original to cropped space
                    cr = crop_result.crop_region
                    task_kw["analysis_fragment"] = {
                        "x": analysis_fragment["x"] - cr["x"],
                        "y": analysis_fragment["y"] - cr["y"],
                        "width": analysis_fragment["width"],
                        "height": analysis_fragment["height"],
                    }
                elif resolution is not None:
                    actual_source = _get_source(resolution)
                    task_kw["source_image"] = str(actual_source)
                # else: source_image already points to the original

                record = _encode_and_measure(
                    project_root_str=project_root_str,
                    **task_kw,
                )
                all_records.append(record)

    return all_records


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
    crop: int | None = None,
    analysis_fragment: dict[str, int] | None = None,
    crop_region: dict[str, int] | None = None,
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

    When *analysis_fragment* is provided (crop-impact mode), quality
    metrics are measured only on the extracted fragment while
    ``width`` / ``height`` reflect the full cropped image dimensions
    (for correct bytes-per-pixel calculation).

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
        crop: Crop tag (target longest-edge if cropped).
        analysis_fragment: Fragment coordinates in the *source image*
            coordinate space.  When set, quality metrics are measured
            only on this region.
        crop_region: The crop window applied to the original image
            (in original image coordinates).  Stored in the record for
            downstream comparison use.
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
        crop=crop,
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
                return _error_record(
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
                    crop=crop,
                    analysis_fragment=analysis_fragment,
                    crop_region=crop_region,
                )
        except Exception as exc:
            return _error_record(
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
                crop=crop,
                analysis_fragment=analysis_fragment,
                crop_region=crop_region,
            )

        encoding_time = time.monotonic() - t0

        if not result.success or result.output_path is None:
            return _error_record(
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
                crop=crop,
                analysis_fragment=analysis_fragment,
                crop_region=crop_region,
            )

        file_size = result.file_size or result.output_path.stat().st_size

        # --- Measure ----------------------------------------------------------
        try:
            measurer = QualityMeasurer()
            if analysis_fragment is not None:
                # Crop-impact mode: measure quality only on the fragment
                from src.quality import extract_fragment as _extract_frag

                frag_dir = Path(tmpdir) / "fragments"
                frag_dir.mkdir(exist_ok=True)
                src_frag = _extract_frag(
                    source_path, analysis_fragment, frag_dir / "src_frag.png"
                )
                enc_frag = _extract_frag(
                    result.output_path, analysis_fragment, frag_dir / "enc_frag.png"
                )
                metrics = measurer.measure_all(src_frag, enc_frag)
            else:
                metrics = measurer.measure_all(
                    source_path,
                    result.output_path,
                )
        except Exception as exc:
            return _error_record(
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
                crop=crop,
                analysis_fragment=analysis_fragment,
                crop_region=crop_region,
            )

        # --- Optionally persist artifact --------------------------------------
        encoded_path_label = ""
        if save_dir_str is not None:
            save_dir = Path(save_dir_str)
            save_dir.mkdir(parents=True, exist_ok=True)
            dest = save_dir / result.output_path.name
            shutil.copy2(result.output_path, dest)
            encoded_path_label = _make_rel(dest, project_root)

    # tmpdir auto-cleaned here
    return QualityRecord(
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
        crop=crop,
        analysis_fragment=analysis_fragment,
        crop_region=crop_region,
        extra_args=extra_args,
        measurement_error=metrics.error_message,
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
    crop: int | None = None,
    analysis_fragment: dict[str, int] | None = None,
    crop_region: dict[str, int] | None = None,
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
        crop=crop,
        analysis_fragment=analysis_fragment,
        crop_region=crop_region,
        extra_args=extra_args,
        measurement_error=error,
    )


# ---------------------------------------------------------------------------
# Task generation helpers
# ---------------------------------------------------------------------------


def _expand_encoder_tasks(
    source_image: Path,
    original_image: Path,
    enc: EncoderConfig,
    save_artifact_dir: Path | None,
    study_id: str,
    analysis_fragment: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Expand one :class:`EncoderConfig` into keyword-arg dicts for
    :func:`_encode_and_measure`.

    Resolution and crop are part of the Cartesian product alongside
    quality, chroma, speed, effort, and method.  They are mutually
    exclusive on the same encoder: if both are set a :class:`ValueError`
    is raised.

    When *analysis_fragment* is set (crop-impact mode), the fragment
    coordinates and crop region are included in each task dict so that
    :func:`_encode_and_measure` can measure metrics on the fragment only.

    Returns a list of dicts that can be unpacked as
    ``_encode_and_measure(**kw)`` (all values are pickle-safe).
    """
    if enc.resolution and enc.crop:
        msg = (
            f"Encoder '{enc.format}' has both 'resolution' and 'crop' "
            f"set — they are mutually exclusive."
        )
        raise ValueError(msg)

    chroma_options: list[str | None] = (
        list(enc.chroma_subsampling) if enc.chroma_subsampling else [None]
    )
    speed_options: list[int | None] = list(enc.speed) if enc.speed else [None]
    effort_options: list[int | None] = list(enc.effort) if enc.effort else [None]
    method_options: list[int | None] = list(enc.method) if enc.method else [None]
    resolution_options: list[int | None] = list(enc.resolution) if enc.resolution else [None]
    crop_options: list[int | None] = list(enc.crop) if enc.crop else [None]

    tasks: list[dict[str, Any]] = []
    for resolution in resolution_options:
        for crop_level in crop_options:
            if resolution is not None:
                level_label = f"r{resolution}"
            elif crop_level is not None:
                level_label = f"c{crop_level}"
            else:
                level_label = "original"

            if save_artifact_dir is not None:
                save_dir_str: str | None = str(save_artifact_dir / enc.format / level_label)
            else:
                save_dir_str = None

            # Build source_image_label for preprocessed images
            if resolution is not None:
                source_image_label: str | None = (
                    f"data/preprocessed/{study_id}/r{resolution}/{source_image.stem}_r{resolution}.png"
                )
            elif crop_level is not None:
                source_image_label = (
                    f"data/preprocessed/{study_id}/c{crop_level}/{source_image.stem}_c{crop_level}.png"
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
                                        "crop": crop_level,
                                        "analysis_fragment": analysis_fragment,
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
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        config: StudyConfig,
        *,
        time_budget: float | None = None,
        save_artifacts: bool = False,
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
        all_crops: set[int] = set()
        for enc in config.encoders:
            if enc.resolution:
                all_resolutions.update(enc.resolution)
            if enc.crop:
                all_crops.update(enc.crop)

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
        if all_crops:
            crop_labels = sorted(f"c{c}" for c in all_crops)
            frag_sz = config.analysis_fragment_size or 200
            print(f"Crop levels: {', '.join(crop_labels)}")
            print(f"Analysis fragment: {frag_sz}x{frag_sz}")
        print(f"Workers: {num_workers}")
        if save_artifacts:
            print(f"Saving artifacts to: {save_artifact_dir}")
        print()

        # --- Prepare config dict for serialization ----------------------------
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
                    "crop": enc.crop,
                    "extra_args": enc.extra_args,
                }
                for enc in config.encoders
            ],
            "analysis_fragment_size": config.analysis_fragment_size,
            "crop_too_small_strategy": config.crop_too_small_strategy,
        }

        # --- Process images with worker-per-image model -----------------------
        # Each worker processes one complete image (all tasks sequentially),
        # then pulls the next image from the queue. This keeps workers fully
        # utilized and staggers memory-intensive operations.
        all_records: list[QualityRecord] = []
        start_time = time.monotonic()
        images_processed = 0

        mp_ctx = multiprocessing.get_context("spawn")

        save_artifact_dir_str = str(save_artifact_dir) if save_artifact_dir else None

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
                    save_artifact_dir_str=save_artifact_dir_str,
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
                    image_records = future.result()

                    all_records.extend(image_records)
                    images_processed += 1

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
                                save_artifact_dir_str=save_artifact_dir_str,
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
        )

        # --- Summary ----------------------------------------------------------
        total_errors = sum(1 for r in all_records if r.measurement_error is not None)
        print(f"\nPipeline complete in {_format_duration(elapsed_total)}")
        print(f"  Images processed: {images_processed}/{len(all_images)}")
        print(f"  Total measurements: {len(all_records)}")
        if total_errors:
            print(f"  Errors: {total_errors}")

        return results
