"""Quality measurement module.

This module handles measuring image quality using various metrics
including SSIMULACRA2, PSNR, SSIM, and Butteraugli. It also provides
utilities for loading encoding results and running batch measurements.

Quality measurement tools often have limited format support, so encoded
images are converted to PNG before measurement. To minimize disk IO,
temporary files are stored in memory-backed storage (/dev/shm on Linux)
when available.

Important: Quality measurements compare encoded images against their
source_image (the preprocessed version used for encoding), NOT the
original_image. This ensures images are compared at the same resolution,
which is required by all quality measurement tools.
"""

import json
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _get_tmpdir() -> str | None:
    """Get the best temporary directory for storing converted images.

    Returns /dev/shm (memory-backed) on Linux systems for faster IO,
    or None to use the system default.
    """
    shm_path = Path("/dev/shm")
    if shm_path.exists() and shm_path.is_dir():
        return str(shm_path)
    return None


@dataclass
class QualityMetrics:
    """Container for quality measurement results."""

    ssimulacra2: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    butteraugli: float | None = None
    error_message: str | None = None


class QualityMeasurer:
    """Handles quality measurements for encoded images."""

    @staticmethod
    def _to_png(image_path: Path, output_path: Path) -> None:
        """Convert an image to PNG format for measurement tools.

        Many quality measurement tools have limited format support (e.g.,
        ssimulacra2 and butteraugli can't read AVIF or JXL directly).
        This method converts any image format to PNG.

        For most formats, Pillow is used. For formats Pillow doesn't support
        (like JPEG XL), format-specific decoders are used.

        Args:
            image_path: Path to the source image (any format)
            output_path: Path where PNG will be written

        Raises:
            IOError: If image cannot be read or written
            subprocess.CalledProcessError: If external decoder fails
        """
        from PIL import Image

        # Handle JPEG XL separately since Pillow doesn't support it
        if image_path.suffix.lower() in (".jxl", ".jpegxl"):
            try:
                cmd = ["djxl", str(image_path), str(output_path)]
                subprocess.run(cmd, capture_output=True, check=True)
                return
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                msg = f"Failed to decode JXL file {image_path}: {e}"
                raise OSError(msg) from e

        # Handle AVIF separately since Pillow support may be limited
        if image_path.suffix.lower() == ".avif":
            try:
                # Try avifdec if available
                cmd = ["avifdec", str(image_path), str(output_path)]
                subprocess.run(cmd, capture_output=True, check=True)
                return
            except FileNotFoundError:
                # Fall back to Pillow if avifdec not available
                pass

        # Use Pillow for all other formats
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (some formats use different color modes)
                if img.mode not in ("RGB", "L"):
                    converted_img = img.convert("RGB")
                    converted_img.save(output_path, format="PNG")
                else:
                    img.save(output_path, format="PNG")
        except Exception as e:
            msg = f"Failed to convert {image_path} to PNG: {e}"
            raise OSError(msg) from e

    def measure_ssimulacra2(self, original: Path, compressed: Path) -> float | None:
        """Measure SSIMULACRA2 score between two images.

        Both images are converted to PNG if needed, as ssimulacra2 has
        limited format support. Uses memory-backed temp storage for speed.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            SSIMULACRA2 score (higher is better, 100 = lossless)
        """
        try:
            with tempfile.TemporaryDirectory(dir=_get_tmpdir()) as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Convert both images to PNG if they're not already PNG
                orig_for_measure = original
                comp_for_measure = compressed

                if original.suffix.lower() != ".png":
                    orig_for_measure = tmpdir_path / "original.png"
                    self._to_png(original, orig_for_measure)

                if compressed.suffix.lower() != ".png":
                    comp_for_measure = tmpdir_path / "compressed.png"
                    self._to_png(compressed, comp_for_measure)

                cmd = ["ssimulacra2", str(orig_for_measure), str(comp_for_measure)]
                result = subprocess.run(cmd, capture_output=True, check=True, text=True)

                # Parse the output to extract the score
                output = result.stdout.strip()
                try:
                    return float(output)
                except ValueError:
                    # If direct parsing fails, try to extract from formatted output
                    if ":" in output:
                        score_str = output.split(":")[-1].strip()
                        return float(score_str)
                return None
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"SSIMULACRA2 measurement failed: {e}")
            return None

    def measure_psnr(self, original: Path, compressed: Path) -> float | None:
        """Measure PSNR between two images using FFmpeg.

        FFmpeg has good format support, but we convert to PNG for consistency.
        Uses memory-backed temp storage for speed.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            PSNR value in dB (higher is better)
        """
        try:
            with tempfile.TemporaryDirectory(dir=_get_tmpdir()) as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Convert both images to PNG if they're not already PNG
                orig_for_measure = original
                comp_for_measure = compressed

                if original.suffix.lower() != ".png":
                    orig_for_measure = tmpdir_path / "original.png"
                    self._to_png(original, orig_for_measure)

                if compressed.suffix.lower() != ".png":
                    comp_for_measure = tmpdir_path / "compressed.png"
                    self._to_png(compressed, comp_for_measure)

                cmd = [
                    "ffmpeg",
                    "-i",
                    str(comp_for_measure),
                    "-i",
                    str(orig_for_measure),
                    "-lavfi",
                    "psnr",
                    "-f",
                    "null",
                    "-",
                ]
                result = subprocess.run(cmd, capture_output=True, check=True, text=True)
                # Parse PSNR from stderr (FFmpeg outputs to stderr)
                for line in result.stderr.split("\n"):
                    if "average:" in line.lower():
                        # Format: "... average:XX.XX ..."
                        parts = line.split("average:")
                        if len(parts) > 1:
                            value_str = parts[1].split()[0]
                            return float(value_str)
                return None
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"PSNR measurement failed: {e}")
            return None

    def measure_ssim(self, original: Path, compressed: Path) -> float | None:
        """Measure SSIM between two images using FFmpeg.

        FFmpeg has good format support, but we convert to PNG for consistency.
        Uses memory-backed temp storage for speed.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            SSIM value (0-1, higher is better)
        """
        try:
            with tempfile.TemporaryDirectory(dir=_get_tmpdir()) as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Convert both images to PNG if they're not already PNG
                orig_for_measure = original
                comp_for_measure = compressed

                if original.suffix.lower() != ".png":
                    orig_for_measure = tmpdir_path / "original.png"
                    self._to_png(original, orig_for_measure)

                if compressed.suffix.lower() != ".png":
                    comp_for_measure = tmpdir_path / "compressed.png"
                    self._to_png(compressed, comp_for_measure)

                cmd = [
                    "ffmpeg",
                    "-i",
                    str(comp_for_measure),
                    "-i",
                    str(orig_for_measure),
                    "-lavfi",
                    "ssim",
                    "-f",
                    "null",
                    "-",
                ]
                result = subprocess.run(cmd, capture_output=True, check=True, text=True)
                # Parse SSIM from stderr
                for line in result.stderr.split("\n"):
                    if "all:" in line.lower():
                        # Format: "... All:X.XXXXXX ..."
                        parts = line.split("All:")
                        if len(parts) > 1:
                            value_str = parts[1].split()[0]
                            return float(value_str)
                return None
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"SSIM measurement failed: {e}")
            return None

    def measure_butteraugli(self, original: Path, compressed: Path) -> float | None:
        """Measure Butteraugli distance between two images.

        Both images are converted to PNG if needed, as butteraugli has
        limited format support. Uses memory-backed temp storage for speed.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            Butteraugli distance (lower is better, <1.0 = excellent)
        """
        try:
            with tempfile.TemporaryDirectory(dir=_get_tmpdir()) as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Convert both images to PNG if they're not already PNG
                orig_for_measure = original
                comp_for_measure = compressed

                if original.suffix.lower() != ".png":
                    orig_for_measure = tmpdir_path / "original.png"
                    self._to_png(original, orig_for_measure)

                if compressed.suffix.lower() != ".png":
                    comp_for_measure = tmpdir_path / "compressed.png"
                    self._to_png(compressed, comp_for_measure)

                cmd = ["butteraugli_main", str(orig_for_measure), str(comp_for_measure)]
                result = subprocess.run(cmd, capture_output=True, check=True, text=True)

                # Parse the output to extract the distance
                output = result.stdout.strip()
                # Try to find a floating point number in the output
                import re

                match = re.search(r"[\d.]+", output)
                if match:
                    return float(match.group())
                return None
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Butteraugli measurement failed: {e}")
            return None

    def measure_all(self, original: Path, compressed: Path) -> QualityMetrics:
        """Measure all available quality metrics.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            QualityMetrics object with all measured values
        """
        return QualityMetrics(
            ssimulacra2=self.measure_ssimulacra2(original, compressed),
            psnr=self.measure_psnr(original, compressed),
            ssim=self.measure_ssim(original, compressed),
            butteraugli=self.measure_butteraugli(original, compressed),
        )


@dataclass
class EncodingRecord:
    """Record of a single encoding from the encoding results JSON.

    Attributes:
        source_image: Path to the preprocessed image used for encoding.
            This is the actual input to the encoder and is used as the
            reference for quality measurements (same resolution as encoded).
        original_image: Path to the original unprocessed image from the dataset.
            This is tracked for provenance but not used for quality measurement
            (may have different resolution than encoded image).
        encoded_path: Path to the encoded output file.
        format: Image format (jpeg, webp, avif, jxl).
        quality: Quality parameter used for encoding (0-100).
        file_size: Size of the encoded file in bytes.
        width: Width of the encoded image in pixels.
        height: Height of the encoded image in pixels.
        source_file_size: Size of the source image file in bytes.
        chroma_subsampling: Chroma subsampling mode if applicable.
        speed: Encoding speed parameter if applicable.
        resolution: Resolution tag for preprocessed images (e.g., 1920).
        extra_args: Additional encoder-specific arguments.
    """

    source_image: str
    original_image: str
    encoded_path: str
    format: str
    quality: int
    file_size: int
    width: int
    height: int
    source_file_size: int
    chroma_subsampling: str | None = None
    speed: int | None = None
    resolution: int | None = None
    extra_args: dict[str, str | int | bool] | None = None


@dataclass
class EncodingResults:
    """Container for encoding study results."""

    study_id: str
    study_name: str
    dataset: dict[str, Any]
    encodings: list[EncodingRecord]
    timestamp: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> "EncodingResults":
        """Load encoding results from a JSON file.

        Args:
            path: Path to the encoding results JSON file

        Returns:
            EncodingResults instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the JSON is invalid or missing required fields
        """
        if not path.exists():
            msg = f"Encoding results file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in {path}: {e}"
            raise ValueError(msg) from e

        # Validate required fields
        required_fields = ["study_id", "study_name", "dataset", "encodings"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            msg = f"Missing required fields in {path}: {missing}"
            raise ValueError(msg)

        encodings = [EncodingRecord(**rec) for rec in data["encodings"]]

        return cls(
            study_id=data["study_id"],
            study_name=data["study_name"],
            dataset=data["dataset"],
            encodings=encodings,
            timestamp=data.get("timestamp"),
        )


@dataclass
class QualityRecord:
    """Quality measurement record for a single encoding."""

    source_image: str
    original_image: str
    encoded_path: str
    format: str
    quality: int
    file_size: int
    width: int
    height: int
    source_file_size: int
    ssimulacra2: float | None
    psnr: float | None
    ssim: float | None
    butteraugli: float | None
    chroma_subsampling: str | None = None
    speed: int | None = None
    resolution: int | None = None
    extra_args: dict[str, str | int | bool] | None = None
    measurement_error: str | None = None


@dataclass
class QualityResults:
    """Container for quality measurement results."""

    study_id: str
    study_name: str
    dataset: dict[str, Any]
    measurements: list[QualityRecord]
    timestamp: str
    encoding_timestamp: str | None = None

    def save(self, path: Path) -> None:
        """Save quality results to a JSON file.

        Args:
            path: Path where the JSON file will be written
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "study_id": self.study_id,
            "study_name": self.study_name,
            "dataset": self.dataset,
            "encoding_timestamp": self.encoding_timestamp,
            "timestamp": self.timestamp,
            "measurements": [
                {
                    "source_image": rec.source_image,
                    "original_image": rec.original_image,
                    "encoded_path": rec.encoded_path,
                    "format": rec.format,
                    "quality": rec.quality,
                    "file_size": rec.file_size,
                    "width": rec.width,
                    "height": rec.height,
                    "source_file_size": rec.source_file_size,
                    "ssimulacra2": rec.ssimulacra2,
                    "psnr": rec.psnr,
                    "ssim": rec.ssim,
                    "butteraugli": rec.butteraugli,
                    "chroma_subsampling": rec.chroma_subsampling,
                    "speed": rec.speed,
                    "resolution": rec.resolution,
                    "extra_args": rec.extra_args,
                    "measurement_error": rec.measurement_error,
                }
                for rec in self.measurements
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _execute_measurement_task(
    encoding: EncodingRecord,
    project_root: Path,
) -> tuple[EncodingRecord, QualityRecord]:
    """Execute quality measurements for a single encoding (top-level function for multiprocessing).

    Args:
        encoding: The encoding record to measure
        project_root: Project root for resolving relative paths

    Returns:
        Tuple of (encoding, quality_record)
    """
    measurer = QualityMeasurer()

    # Use source_image as reference (preprocessed version) instead of original_image
    # This ensures we compare images of the same resolution
    reference_path = project_root / encoding.source_image
    encoded_path = project_root / encoding.encoded_path

    error_message = None
    if not reference_path.exists():
        error_message = f"Reference image not found: {reference_path}"
    elif not encoded_path.exists():
        error_message = f"Encoded image not found: {encoded_path}"

    if error_message:
        return (
            encoding,
            QualityRecord(
                source_image=encoding.source_image,
                original_image=encoding.original_image,
                encoded_path=encoding.encoded_path,
                format=encoding.format,
                quality=encoding.quality,
                file_size=encoding.file_size,
                width=encoding.width,
                height=encoding.height,
                source_file_size=encoding.source_file_size,
                ssimulacra2=None,
                psnr=None,
                ssim=None,
                butteraugli=None,
                chroma_subsampling=encoding.chroma_subsampling,
                speed=encoding.speed,
                resolution=encoding.resolution,
                extra_args=encoding.extra_args,
                measurement_error=error_message,
            ),
        )

    metrics = measurer.measure_all(reference_path, encoded_path)

    return (
        encoding,
        QualityRecord(
            source_image=encoding.source_image,
            original_image=encoding.original_image,
            encoded_path=encoding.encoded_path,
            format=encoding.format,
            quality=encoding.quality,
            file_size=encoding.file_size,
            width=encoding.width,
            height=encoding.height,
            source_file_size=encoding.source_file_size,
            ssimulacra2=metrics.ssimulacra2,
            psnr=metrics.psnr,
            ssim=metrics.ssim,
            butteraugli=metrics.butteraugli,
            chroma_subsampling=encoding.chroma_subsampling,
            speed=encoding.speed,
            resolution=encoding.resolution,
            extra_args=encoding.extra_args,
            measurement_error=metrics.error_message,
        ),
    )


class QualityMeasurementRunner:
    """Runs quality measurements on encoding results."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the measurement runner.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root

    def run(
        self, encoding_results: EncodingResults, num_workers: int | None = None
    ) -> QualityResults:
        """Run quality measurements on all encodings.

        Args:
            encoding_results: Encoding results to measure
            num_workers: Number of parallel workers (default: CPU count)

        Returns:
            QualityResults with all measurements
        """
        encodings = encoding_results.encodings
        total = len(encodings)

        print(f"Measuring quality for {total} encodings...")
        print(f"Study: {encoding_results.study_name}")
        print(f"Dataset: {encoding_results.dataset['id']}")
        print()

        measurements: list[QualityRecord] = []
        completed = 0
        start_time = time.monotonic()

        if num_workers is None:
            import multiprocessing

            num_workers = multiprocessing.cpu_count()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_execute_measurement_task, enc, self.project_root): enc
                for enc in encodings
            }

            for future in as_completed(futures):
                _, record = future.result()
                measurements.append(record)
                completed += 1

                elapsed = time.monotonic() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0

                # Progress message
                ssim2_str = f"{record.ssimulacra2:.2f}" if record.ssimulacra2 else "N/A"
                print(
                    f"[{completed}/{total}] {record.format.upper()} q{record.quality} "
                    f"| {record.encoded_path.split('/')[-1][:30]:30} "
                    f"| SSIM2: {ssim2_str:>6} "
                    f"| ETA: {self._format_duration(eta)}"
                )

        elapsed_total = time.monotonic() - start_time
        print()
        print(f"Measurement complete in {self._format_duration(elapsed_total)}")
        print(f"  Measured: {len(measurements)}")
        print(f"  Average rate: {completed / elapsed_total:.1f} measurements/sec")

        # Count failed measurements
        failed = sum(1 for m in measurements if m.measurement_error is not None)
        if failed > 0:
            print(f"  Failed: {failed}")

        return QualityResults(
            study_id=encoding_results.study_id,
            study_name=encoding_results.study_name,
            dataset=encoding_results.dataset,
            measurements=measurements,
            timestamp=datetime.now(UTC).isoformat(),
            encoding_timestamp=encoding_results.timestamp,
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration in seconds to a human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string like "1h 23m 45s" or "5m 12s"
        """
        if seconds < 0:
            seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"
