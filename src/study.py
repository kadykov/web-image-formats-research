"""Study configuration and execution module.

This module handles loading study configurations from JSON files,
expanding parameter sweeps into concrete encoding tasks, orchestrating
the preprocessing and encoding pipeline, and writing structured results.
"""

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.dataset import DatasetFetcher
from src.encoder import ImageEncoder
from src.preprocessing import ImagePreprocessor


@dataclass
class EncoderConfig:
    """Configuration for a single encoder sweep."""

    format: str
    quality: list[int]
    chroma_subsampling: list[str] | None = None
    speed: list[int] | None = None
    effort: list[int] | None = None
    method: list[int] | None = None
    extra_args: dict[str, str | int | bool] | None = None


@dataclass
class StudyConfig:
    """Configuration for a complete encoding study."""

    id: str
    name: str
    dataset_id: str
    max_images: int | None
    encoders: list[EncoderConfig]
    resize: list[int] | None = None
    description: str | None = None
    time_budget: float | None = None

    @classmethod
    def from_file(cls, config_path: Path) -> "StudyConfig":
        """Load a study configuration from a JSON file.

        Args:
            config_path: Path to the study JSON file

        Returns:
            StudyConfig instance

        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config file has invalid content
        """
        if not config_path.exists():
            msg = f"Study config not found: {config_path}"
            raise FileNotFoundError(msg)

        with open(config_path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudyConfig":
        """Create StudyConfig from a dictionary.

        Args:
            data: Dictionary matching the study JSON schema

        Returns:
            StudyConfig instance

        Raises:
            ValueError: If required fields are missing
        """
        if "id" not in data or "dataset" not in data or "encoders" not in data:
            msg = "Study config must have 'id', 'dataset', and 'encoders' fields"
            raise ValueError(msg)

        encoders = [_parse_encoder_config(enc) for enc in data["encoders"]]

        preprocessing = data.get("preprocessing", {})
        resize = preprocessing.get("resize")

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            dataset_id=data["dataset"]["id"],
            max_images=data["dataset"].get("max_images"),
            encoders=encoders,
            resize=resize,
            description=data.get("description"),
            time_budget=data.get("time_budget"),
        )


def _parse_quality(quality: int | list[int] | dict[str, int]) -> list[int]:
    """Parse quality parameter into a list of integers.

    Supports:
    - Single integer: ``75`` → ``[75]``
    - Explicit list: ``[60, 75, 90]`` → ``[60, 75, 90]``
    - Range object: ``{"start": 30, "stop": 90, "step": 10}`` → ``[30, 40, ..., 90]``

    Args:
        quality: Quality specification from the study config

    Returns:
        List of quality values
    """
    if isinstance(quality, int):
        return [quality]
    if isinstance(quality, list):
        return quality
    if isinstance(quality, dict):
        return list(range(quality["start"], quality["stop"] + 1, quality["step"]))
    msg = f"Invalid quality specification: {quality}"
    raise ValueError(msg)


def _parse_encoder_config(data: dict[str, Any]) -> EncoderConfig:
    """Parse an encoder configuration dictionary into an EncoderConfig.

    Args:
        data: Encoder dictionary from the study JSON

    Returns:
        EncoderConfig instance
    """
    quality = _parse_quality(data["quality"])

    speed_raw = data.get("speed")
    speed: list[int] | None = None
    if speed_raw is not None:
        speed = [speed_raw] if isinstance(speed_raw, int) else speed_raw

    effort_raw = data.get("effort")
    effort: list[int] | None = None
    if effort_raw is not None:
        effort = [effort_raw] if isinstance(effort_raw, int) else effort_raw

    method_raw = data.get("method")
    method: list[int] | None = None
    if method_raw is not None:
        method = [method_raw] if isinstance(method_raw, int) else method_raw

    chroma = data.get("chroma_subsampling")

    return EncoderConfig(
        format=data["format"],
        quality=quality,
        chroma_subsampling=chroma,
        speed=speed,
        effort=effort,
        method=method,
        extra_args=data.get("extra_args"),
    )


@dataclass
class EncodingTask:
    """A single concrete encoding task expanded from the study config."""

    source_image: Path
    original_image: Path
    format: str
    quality: int
    megapixels: float
    chroma_subsampling: str | None = None
    speed: int | None = None
    effort: int | None = None
    method: int | None = None
    resolution: int | None = None
    extra_args: dict[str, str | int | bool] | None = None


@dataclass
class EncodingRecord:
    """Record of a completed encoding, written to the results JSON."""

    source_image: str
    original_image: str
    encoded_path: str
    format: str
    quality: int
    file_size: int
    width: int
    height: int
    source_file_size: int
    encoding_time: float
    chroma_subsampling: str | None = None
    speed: int | None = None
    effort: int | None = None
    method: int | None = None
    resolution: int | None = None
    extra_args: dict[str, str | int | bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        d: dict[str, Any] = {
            "source_image": self.source_image,
            "original_image": self.original_image,
            "encoded_path": self.encoded_path,
            "format": self.format,
            "quality": self.quality,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "source_file_size": self.source_file_size,
            "encoding_time": self.encoding_time,
        }
        if self.chroma_subsampling is not None:
            d["chroma_subsampling"] = self.chroma_subsampling
        if self.speed is not None:
            d["speed"] = self.speed
        if self.effort is not None:
            d["effort"] = self.effort
        if self.method is not None:
            d["method"] = self.method
        if self.resolution is not None:
            d["resolution"] = self.resolution
        if self.extra_args:
            d["extra_args"] = self.extra_args
        return d


@dataclass
class StudyResults:
    """Complete results of a study execution."""

    study_id: str
    study_name: str
    dataset_id: str
    dataset_path: str
    image_count: int
    timestamp: str
    records: list[EncodingRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary matching the encoding-results schema."""
        return {
            "study_id": self.study_id,
            "study_name": self.study_name,
            "dataset": {
                "id": self.dataset_id,
                "path": self.dataset_path,
                "image_count": self.image_count,
            },
            "timestamp": self.timestamp,
            "encodings": [r.to_dict() for r in self.records],
        }

    def save(self, output_path: Path) -> None:
        """Save results to a JSON file.

        Args:
            output_path: Path where the JSON file will be written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {output_path}")


def expand_tasks(
    config: StudyConfig,
    source_images: list[Path],
    original_images: list[Path] | None = None,
    resolution: int | None = None,
) -> list[EncodingTask]:
    """Expand a study config into concrete encoding tasks for a set of images.

    For each source image, generates one task per combination of
    (encoder × quality × chroma_subsampling × speed).

    Args:
        config: Study configuration
        source_images: Images to encode (may be preprocessed copies)
        original_images: Original dataset images (for quality reference).
            If None, source_images are treated as originals.
        resolution: Resolution tag if images were resized

    Returns:
        List of EncodingTask instances
    """
    from PIL import Image

    if original_images is None:
        original_images = source_images

    # Pre-compute megapixels for each source image (avoids reopening per task)
    image_megapixels: dict[Path, float] = {}
    for src in source_images:
        if src not in image_megapixels:
            with Image.open(src) as img:
                w, h = img.size
            image_megapixels[src] = (w * h) / 1_000_000

    tasks: list[EncodingTask] = []
    for src, orig in zip(source_images, original_images, strict=True):
        for enc in config.encoders:
            chroma_options: list[str | None] = (
                list(enc.chroma_subsampling) if enc.chroma_subsampling else [None]
            )
            speed_options: list[int | None] = list(enc.speed) if enc.speed else [None]
            effort_options: list[int | None] = list(enc.effort) if enc.effort else [None]
            method_options: list[int | None] = list(enc.method) if enc.method else [None]

            for q in enc.quality:
                for chroma in chroma_options:
                    for spd in speed_options:
                        for eff in effort_options:
                            for mth in method_options:
                                tasks.append(
                                    EncodingTask(
                                        source_image=src,
                                        original_image=orig,
                                        format=enc.format,
                                        quality=q,
                                        megapixels=image_megapixels[src],
                                        chroma_subsampling=chroma,
                                        speed=spd,
                                        effort=eff,
                                        method=mth,
                                        resolution=resolution,
                                        extra_args=enc.extra_args,
                                    )
                                )
    return tasks


def _interleave_tasks(tasks: list[EncodingTask]) -> list[EncodingTask]:
    """Reorder tasks so that different formats alternate.

    This spreads fast encoders (JPEG, WebP) and slow encoders (AVIF, JXL)
    across the task list so that early timing samples represent a mix of
    encoder speeds, letting the ETA converge faster.

    The interleaving works round-robin across format buckets, preserving
    the original order within each bucket.

    Args:
        tasks: Flat list of encoding tasks (any order)

    Returns:
        Reordered list with formats interleaved
    """
    buckets: dict[str, list[EncodingTask]] = {}
    for t in tasks:
        buckets.setdefault(t.format, []).append(t)

    interleaved: list[EncodingTask] = []
    iterators = {fmt: iter(bucket) for fmt, bucket in buckets.items()}
    while iterators:
        exhausted: list[str] = []
        for fmt, it in iterators.items():
            task = next(it, None)
            if task is not None:
                interleaved.append(task)
            else:
                exhausted.append(fmt)
        for fmt in exhausted:
            del iterators[fmt]
    return interleaved


def _build_output_name(task: EncodingTask, stem: str) -> str:
    """Build a descriptive output filename for an encoding task.

    Args:
        task: The encoding task
        stem: Base filename stem (without extension)

    Returns:
        Filename string (without extension)
    """
    parts = [stem, f"q{task.quality}"]
    if task.chroma_subsampling is not None:
        parts.append(task.chroma_subsampling)
    if task.speed is not None:
        parts.append(f"s{task.speed}")
    if task.effort is not None:
        parts.append(f"e{task.effort}")
    if task.method is not None:
        parts.append(f"m{task.method}")
    if task.resolution is not None:
        parts.append(f"r{task.resolution}")
    return "_".join(parts)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like ``"1h 23m 45s"`` or ``"5m 12s"``
    """
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


class _ETATracker:
    """Estimates remaining time based on megapixels processed per second.

    The tracker accumulates ``(megapixels, elapsed_seconds)`` samples from
    completed tasks, then uses the weighted throughput (MP/s) to predict
    how long the remaining megapixels will take.

    Because different formats encode at vastly different speeds, the
    throughput is tracked per format separately, giving a much more
    accurate ETA when multiple formats are mixed.
    """

    def __init__(self, tasks: list[EncodingTask]) -> None:
        self._start = time.monotonic()
        # Total megapixels remaining per format
        self._remaining_mp: dict[str, float] = {}
        for t in tasks:
            self._remaining_mp[t.format] = self._remaining_mp.get(t.format, 0.0) + t.megapixels
        # Cumulative (megapixels, wall-seconds) per format
        self._processed_mp: dict[str, float] = {}
        self._processed_time: dict[str, float] = {}

    def record(self, task: EncodingTask, elapsed: float) -> None:
        """Record a completed task.

        Args:
            task: The task that finished
            elapsed: Wall-clock seconds the task took
        """
        self._remaining_mp[task.format] -= task.megapixels
        self._processed_mp[task.format] = self._processed_mp.get(task.format, 0.0) + task.megapixels
        self._processed_time[task.format] = self._processed_time.get(task.format, 0.0) + elapsed

    def eta_seconds(self, num_workers: int) -> float | None:
        """Estimate remaining wall-clock seconds.

        Uses per-format throughput (MP/s) to predict remaining encoding
        time and divides by the number of parallel workers.

        Returns:
            Estimated seconds remaining, or ``None`` if not enough data.
        """
        # Need at least one sample from each format that still has work
        total_remaining_seconds = 0.0
        for fmt, remaining_mp in self._remaining_mp.items():
            if remaining_mp <= 0:
                continue
            done_mp = self._processed_mp.get(fmt, 0.0)
            done_time = self._processed_time.get(fmt, 0.0)
            if done_mp <= 0 or done_time <= 0:
                return None  # not enough data yet
            throughput = done_mp / done_time  # MP per (wall-clock) second
            total_remaining_seconds += remaining_mp / throughput
        # Workers run in parallel, so divide by parallelism
        return total_remaining_seconds / max(num_workers, 1)

    def elapsed(self) -> float:
        """Return wall-clock seconds since tracking started."""
        return time.monotonic() - self._start


def _execute_encoding_task(
    task: EncodingTask,
    output_dir: Path,
    project_root: Path,
) -> tuple[EncodingTask, EncodingRecord | str, float]:
    """Execute a single encoding task (top-level function for multiprocessing).

    Args:
        task: The encoding task to execute
        output_dir: Directory where the encoded file will be written
        project_root: Project root for computing relative paths

    Returns:
        Tuple of (task, result, elapsed_seconds) where *result* is an
        ``EncodingRecord`` on success or an error-message string on failure.
    """
    t0 = time.monotonic()
    encoder = ImageEncoder(output_dir)
    output_name = _build_output_name(task, task.source_image.stem)

    if task.format == "jpeg":
        result = encoder.encode_jpeg(task.source_image, task.quality, output_name=output_name)
    elif task.format == "webp":
        method = task.method if task.method is not None else 4
        result = encoder.encode_webp(
            task.source_image,
            task.quality,
            method=method,
            output_name=output_name,
        )
    elif task.format == "avif":
        speed = task.speed if task.speed is not None else 6
        result = encoder.encode_avif(
            task.source_image,
            task.quality,
            speed=speed,
            chroma_subsampling=task.chroma_subsampling,
            output_name=output_name,
        )
    elif task.format == "jxl":
        effort = task.effort if task.effort is not None else 7
        result = encoder.encode_jxl(
            task.source_image,
            task.quality,
            effort=effort,
            output_name=output_name,
        )
    else:
        return (task, f"Unknown format: {task.format}", time.monotonic() - t0)

    elapsed = time.monotonic() - t0

    if not result.success or result.output_path is None:
        msg = (
            f"FAILED: {task.source_image.name} "
            f"({task.format} q{task.quality}): "
            f"{result.error_message}"
        )
        return (task, msg, elapsed)

    def _make_rel(path: Path) -> str:
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)

    record = EncodingRecord(
        source_image=_make_rel(task.source_image),
        original_image=_make_rel(task.original_image),
        encoded_path=_make_rel(result.output_path),
        format=task.format,
        quality=task.quality,
        file_size=result.file_size or 0,
        width=int(task.megapixels * 1_000_000) if task.megapixels > 0 else 0,
        height=0,  # placeholder, filled in below
        source_file_size=task.source_image.stat().st_size,
        encoding_time=elapsed,
        chroma_subsampling=task.chroma_subsampling,
        speed=task.speed,
        effort=task.effort,
        method=task.method,
        resolution=task.resolution,
        extra_args=task.extra_args,
    )
    # Read real dimensions from source image (not the encoded file,
    # because PIL may not support formats like JXL).
    from PIL import Image

    with Image.open(task.source_image) as img:
        record.width, record.height = img.size

    return (task, record, elapsed)


class StudyRunner:
    """Executes an encoding study end-to-end."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the study runner.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.data_dir = project_root / "data"

    def _collect_images(self, dataset_dir: Path, max_images: int | None) -> list[Path]:
        """Collect image paths from a dataset directory.

        Args:
            dataset_dir: Directory containing images
            max_images: Maximum number of images to return, or None for all

        Returns:
            Sorted list of image paths
        """
        extensions = {".png", ".jpg", ".jpeg", ".ppm", ".pgm", ".bmp", ".tiff", ".tif"}
        images = sorted(
            p for p in dataset_dir.rglob("*") if p.suffix.lower() in extensions and p.is_file()
        )
        if max_images is not None:
            images = images[:max_images]
        return images

    def _preprocess_images(
        self,
        images: list[Path],
        resolution: int,
        output_dir: Path,
    ) -> list[Path]:
        """Resize images to a target resolution (longest edge).

        Args:
            images: Source image paths
            resolution: Target longest-edge size in pixels
            output_dir: Directory for preprocessed images

        Returns:
            List of paths to resized images
        """
        preprocessor = ImagePreprocessor(output_dir)
        resized: list[Path] = []
        for img_path in images:
            output_name = f"{img_path.stem}_r{resolution}.png"
            result = preprocessor.resize_image(
                img_path,
                target_size=(resolution, resolution),
                output_name=output_name,
                keep_aspect_ratio=True,
            )
            resized.append(result)
        return resized

    def _make_relative(self, path: Path) -> str:
        """Make a path relative to the project root.

        Args:
            path: Absolute path

        Returns:
            String path relative to project root
        """
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def run(self, config: StudyConfig) -> StudyResults:
        """Execute a complete encoding study.

        Tasks are executed in parallel across CPU cores.  Each encoder is
        forced into single-threaded mode (``-j 1``, ``--num_threads=1``)
        so that parallelism comes from encoding different images
        concurrently.

        Tasks are interleaved by format before submission so that the
        per-format ETA converges quickly.

        Args:
            config: Study configuration

        Returns:
            StudyResults with all encoding records
        """
        # Resolve dataset
        fetcher = DatasetFetcher(self.data_dir / "datasets")
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

        original_images = self._collect_images(dataset_dir, config.max_images)
        if not original_images:
            msg = f"No images found in {dataset_dir}"
            raise FileNotFoundError(msg)

        print(f"Study: {config.name}")
        print(f"Dataset: {config.dataset_id} ({len(original_images)} images)")

        results = StudyResults(
            study_id=config.id,
            study_name=config.name,
            dataset_id=config.dataset_id,
            dataset_path=self._make_relative(dataset_dir),
            image_count=len(original_images),
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

        # Determine resolutions to process
        resolutions: list[int | None] = [None]  # None means original resolution
        if config.resize:
            resolutions = list(config.resize)

        study_output_dir = self.data_dir / "encoded" / config.id
        max_workers = os.cpu_count() or 1

        for resolution in resolutions:
            if resolution is not None:
                res_label = f"r{resolution}"
                print(f"\nPreprocessing: resizing to {resolution}px (longest edge)...")
                preprocess_dir = self.data_dir / "preprocessed" / config.id / res_label
                source_images = self._preprocess_images(original_images, resolution, preprocess_dir)
            else:
                res_label = "original"
                source_images = original_images

            # Expand and interleave tasks for this resolution
            tasks = expand_tasks(config, source_images, original_images, resolution)
            tasks = _interleave_tasks(tasks)
            total = len(tasks)
            print(f"Resolution {res_label}: {total} encoding tasks ({max_workers} workers)")

            # Pre-create output directories
            output_dirs: dict[str, Path] = {}
            for enc_config in config.encoders:
                fmt = enc_config.format
                encoder_dir = study_output_dir / fmt / res_label
                encoder_dir.mkdir(parents=True, exist_ok=True)
                output_dirs[fmt] = encoder_dir

            # Run encoding tasks in parallel using 'spawn' context to avoid
            # fork-safety issues on Python 3.13+ in threaded environments.
            eta_tracker = _ETATracker(tasks)
            completed = 0
            mp_ctx = multiprocessing.get_context("spawn")

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
                futures = {
                    executor.submit(
                        _execute_encoding_task,
                        task,
                        output_dirs[task.format],
                        self.project_root,
                    ): task
                    for task in tasks
                }

                for future in as_completed(futures):
                    completed += 1
                    task, result, elapsed = future.result()
                    eta_tracker.record(task, elapsed)

                    if isinstance(result, EncodingRecord):
                        results.records.append(result)
                    else:
                        print(f"  {result}")

                    if completed % 20 == 0 or completed == total:
                        eta = eta_tracker.eta_seconds(max_workers)
                        elapsed_total = eta_tracker.elapsed()
                        if eta is not None:
                            print(
                                f"  {res_label}: {completed}/{total} done "
                                f"[{_format_duration(elapsed_total)} elapsed, "
                                f"ETA {_format_duration(eta)}]"
                            )
                        else:
                            print(
                                f"  {res_label}: {completed}/{total} done "
                                f"[{_format_duration(elapsed_total)} elapsed]"
                            )

        print(f"\nCompleted: {len(results.records)} successful encodings")
        return results
