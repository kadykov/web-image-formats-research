"""Merged encode+measure pipeline with time-budget support.

This module provides a unified pipeline that processes images one-by-one,
encoding and measuring quality in a single pass. This eliminates the need
to store intermediate encoded files on disk and allows time-budget-based
processing where the pipeline processes as many images as possible within
a given time constraint.

Key advantages over the separate encode → measure workflow:
- **Time-budget control**: Set a wall-clock time limit instead of guessing
  how many images to process. The pipeline processes as many images as
  possible within the budget.
- **Reduced disk IO**: Encoded files are written to temporary storage
  and cleaned up after measurement. Optional ``save_artifacts`` flag
  persists them to disk.
- **Per-image error isolation**: All external tool operations (encoding +
  measurement) for one image are grouped. If encoding or measurement
  fails for an image, the pipeline logs the error and moves on.
- **Deterministic runtime**: The user knows upfront how long the pipeline
  will run (the time budget), and the pipeline maximises the number of
  images processed within it.
"""

import multiprocessing
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Atomic encode-and-measure function (top-level for multiprocessing)
# ---------------------------------------------------------------------------


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
            )

        file_size = result.file_size or result.output_path.stat().st_size

        # --- Measure ----------------------------------------------------------
        try:
            measurer = QualityMeasurer()
            metrics = measurer.measure_all(source_path, result.output_path)
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
# Task generation helpers
# ---------------------------------------------------------------------------


def _expand_encoder_tasks(
    source_image: Path,
    original_image: Path,
    enc: EncoderConfig,
    resolution: int | None,
    save_artifact_dir: Path | None,
    source_image_label: str | None,
) -> list[dict[str, Any]]:
    """Expand one :class:`EncoderConfig` into keyword-arg dicts for
    :func:`_encode_and_measure`.

    Returns a list of dicts that can be unpacked as
    ``_encode_and_measure(**kw)`` (all values are pickle-safe).
    """
    chroma_options: list[str | None] = (
        list(enc.chroma_subsampling) if enc.chroma_subsampling else [None]
    )
    speed_options: list[int | None] = list(enc.speed) if enc.speed else [None]
    effort_options: list[int | None] = list(enc.effort) if enc.effort else [None]
    method_options: list[int | None] = list(enc.method) if enc.method else [None]

    res_label = f"r{resolution}" if resolution else "original"

    if save_artifact_dir is not None:
        save_dir_str: str | None = str(save_artifact_dir / enc.format / res_label)
    else:
        save_dir_str = None

    tasks: list[dict[str, Any]] = []
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

    Processes dataset images one-by-one. For each image the runner:

    1. Preprocesses (resize) for every configured resolution.
    2. Encodes all parameter combinations in parallel.
    3. Measures quality of each encoded variant.
    4. Checks the time budget before starting the next image.

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

        # --- Resolutions ------------------------------------------------------
        resolutions: list[int | None] = [None]  # None = original
        if config.resize:
            resolutions = list(config.resize)

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
        res_labels = [f"r{r}" if r else "original" for r in resolutions]
        print(f"Resolutions: {', '.join(res_labels)}")
        print(f"Workers: {num_workers}")
        if save_artifacts:
            print(f"Saving artifacts to: {save_artifact_dir}")
        print()

        # --- Process images one-by-one ----------------------------------------
        all_records: list[QualityRecord] = []
        start_time = time.monotonic()
        images_processed = 0

        mp_ctx = multiprocessing.get_context("spawn")

        with (
            tempfile.TemporaryDirectory() as prep_tmpdir,
            ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor,
        ):
            for image_path in all_images:
                # Time-budget gate (after first image)
                if time_budget is not None and images_processed > 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= time_budget:
                        print(
                            f"\nTime budget reached "
                            f"({_format_duration(elapsed)} >= "
                            f"{_format_duration(time_budget)})"
                        )
                        break

                # Build tasks for this image across all resolutions
                image_tasks: list[dict[str, Any]] = []
                prep_files: list[Path] = []

                for resolution in resolutions:
                    if resolution is not None:
                        res_dir = Path(prep_tmpdir) / f"r{resolution}"
                        source = self._preprocess_image(image_path, resolution, res_dir)
                        prep_files.append(source)
                        # Label for the output record (stable, not a temp path)
                        src_label: str | None = (
                            f"data/preprocessed/{config.id}"
                            f"/r{resolution}/{image_path.stem}_r{resolution}.png"
                        )
                    else:
                        source = image_path
                        src_label = None  # will default to relative path

                    for enc in config.encoders:
                        image_tasks.extend(
                            _expand_encoder_tasks(
                                source_image=source,
                                original_image=image_path,
                                enc=enc,
                                resolution=resolution,
                                save_artifact_dir=save_artifact_dir,
                                source_image_label=src_label,
                            )
                        )

                # Submit all tasks to the pool
                total = len(image_tasks)
                futures = {
                    executor.submit(
                        _encode_and_measure,
                        project_root_str=str(self.project_root),
                        **kw,
                    ): kw
                    for kw in image_tasks
                }

                # Collect results
                image_records: list[QualityRecord] = []
                for future in as_completed(futures):
                    record = future.result()
                    image_records.append(record)

                all_records.extend(image_records)
                images_processed += 1

                # Progress
                elapsed = time.monotonic() - start_time
                errors = sum(1 for r in image_records if r.measurement_error is not None)
                err_str = f" | {errors} errors" if errors else ""
                budget_str = ""
                if time_budget is not None:
                    remaining = max(0.0, time_budget - elapsed)
                    budget_str = f" | {_format_duration(remaining)} remaining"
                print(
                    f"  [{images_processed}/{len(all_images)}] "
                    f"{image_path.name}: {total} tasks{err_str} "
                    f"[{_format_duration(elapsed)} elapsed{budget_str}]"
                )

                # Clean up preprocessed files for this image
                for pf in prep_files:
                    if pf.exists():
                        pf.unlink()

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
