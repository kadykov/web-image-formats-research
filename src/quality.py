"""Quality measurement module.

This module handles measuring image quality using various metrics
including SSIMULACRA2, PSNR, SSIM, and Butteraugli.

Quality measurement tools often have limited format support, so encoded
images are converted to PNG before measurement.

Important: Quality measurements compare encoded images against their
source_image (the preprocessed version used for encoding), NOT the
original_image. This ensures images are compared at the same resolution,
which is required by all quality measurement tools.
"""

import json
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def get_measurement_tool_version(tool: str) -> str | None:
    """Get version string for a measurement tool.

    Args:
        tool: Name of measurement tool (ssimulacra2, butteraugli, ffmpeg)

    Returns:
        Version string or None if unable to determine
    """
    try:
        if tool == "ssimulacra2":
            # ssimulacra2 doesn't have --version, check if executable exists
            try:
                result = subprocess.run(
                    ["ssimulacra2", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # If command exists, return "available"
                return "available"
            except FileNotFoundError:
                pass

        elif tool == "butteraugli":
            # butteraugli_main typically doesn't have --version
            # Try --help to check if it exists
            try:
                result = subprocess.run(
                    ["butteraugli_main", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # If command exists (even if it fails on missing args), it's available
                return "available"
            except FileNotFoundError:
                pass

        elif tool == "ffmpeg":
            # ffmpeg: "ffmpeg version 6.1.1"
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout
            match = re.search(r"ffmpeg version\s+(\S+)", output)
            if match:
                return match.group(1)
            return "unknown"

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


@dataclass
class QualityMetrics:
    """Container for quality measurement results."""

    ssimulacra2: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    butteraugli: float | None = None
    error_message: str | None = None


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


def read_pfm(pfm_path: Path) -> np.ndarray:
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
    is_color = False
    n_floats = 0

    with open(pfm_path, "rb") as fh:
        magic = fh.readline().strip()
        if magic == b"PF":
            is_color = True
        elif magic == b"Pf":
            is_color = False
        else:
            msg = f"Not a PFM file: unexpected magic {magic!r}"
            raise ValueError(msg)

        dim_line = fh.readline().strip().decode("ascii")
        width, height = (int(x) for x in dim_line.split())

        scale_line = fh.readline().strip().decode("ascii")
        scale = float(scale_line)
        little_endian = scale < 0

        channels = 3 if is_color else 1
        n_floats = width * height * channels
        raw = fh.read(n_floats * 4)

    endian = "<" if little_endian else ">"
    data = np.array(struct.unpack(f"{endian}{n_floats}f", raw), dtype=np.float64)

    if is_color:
        data = data.reshape((height, width, 3)).max(axis=2)
    else:
        data = data.reshape((height, width))

    # PFM rows are stored bottom-to-top; flip to standard orientation
    return np.flipud(data)


def find_worst_region_in_array(
    arr: np.ndarray,
    crop_size: int,
) -> WorstRegion:
    """Find the worst region in a 2-D distortion value array.

    Core sliding-window computation that works with any float array,
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
        limited format support.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            SSIMULACRA2 score (higher is better, 100 = lossless)
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
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

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            PSNR value in dB (higher is better)
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
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

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            SSIM value (0-1, higher is better)
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
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
        limited format support.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            Butteraugli distance (lower is better, <1.0 = excellent)
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
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

    def measure_butteraugli_with_distmap(
        self,
        original: Path,
        compressed: Path,
        distmap_path: Path,
    ) -> float | None:
        """Measure Butteraugli distance and produce a raw distortion map.

        Calls ``butteraugli_main`` with ``--rawdistmap`` to write a PFM
        file containing per-pixel float distortion values, and parses the
        aggregate (3-norm) distance score from stdout.

        This replaces separate calls to :meth:`measure_butteraugli` and
        the comparison module's ``generate_distortion_map``, producing
        both outputs in a single invocation.

        Args:
            original: Path to the original reference image.
            compressed: Path to the compressed/encoded image.
            distmap_path: Path where the raw PFM distortion map will be
                written.  Parent directories are created automatically.

        Returns:
            Butteraugli aggregate distance (lower is better, <1.0 = excellent),
            or ``None`` if the measurement failed.
        """
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
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

                distmap_path.parent.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "butteraugli_main",
                    str(orig_for_measure),
                    str(comp_for_measure),
                    "--distmap",
                    str(tmpdir_path / "distmap_unused.png"),
                    "--rawdistmap",
                    str(distmap_path),
                ]
                result = subprocess.run(cmd, capture_output=True, check=True, text=True)

                # Parse the aggregate distance from stdout
                output = result.stdout.strip()
                match = re.search(r"[\d.]+", output)
                if match:
                    return float(match.group())
                return None
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"Butteraugli measurement failed: {e}")
            return None

    def measure_all(
        self,
        original: Path,
        compressed: Path,
        distmap_path: Path | None = None,
    ) -> QualityMetrics:
        """Measure all available quality metrics.

        When *distmap_path* is provided, the Butteraugli measurement
        additionally produces a raw PFM distortion map at that path.
        This avoids a separate ``butteraugli_main`` invocation later.

        Args:
            original: Path to the original image.
            compressed: Path to the compressed image.
            distmap_path: If given, write a Butteraugli distortion-map
                PFM file here (see :meth:`measure_butteraugli_with_distmap`).

        Returns:
            QualityMetrics object with all measured values.
        """
        if distmap_path is not None:
            butteraugli = self.measure_butteraugli_with_distmap(
                original, compressed, distmap_path,
            )
        else:
            butteraugli = self.measure_butteraugli(original, compressed)

        return QualityMetrics(
            ssimulacra2=self.measure_ssimulacra2(original, compressed),
            psnr=self.measure_psnr(original, compressed),
            ssim=self.measure_ssim(original, compressed),
            butteraugli=butteraugli,
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
    encoding_time: float | None = None
    chroma_subsampling: str | None = None
    speed: int | None = None
    effort: int | None = None
    method: int | None = None
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
    tool_versions: dict[str, str] | None = None
    worst_images: dict[str, dict] | None = None
    worst_fragments: dict[str, dict] | None = None

    def save(self, path: Path) -> None:
        """Save quality results to a JSON file.

        Args:
            path: Path where the JSON file will be written
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "study_id": self.study_id,
            "study_name": self.study_name,
            "dataset": self.dataset,
            "encoding_timestamp": self.encoding_timestamp,
            "timestamp": self.timestamp,
            "tool_versions": self.tool_versions,
            "worst_images": self.worst_images,
            "worst_fragments": self.worst_fragments,
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
                    "encoding_time": rec.encoding_time,
                    "ssimulacra2": rec.ssimulacra2,
                    "psnr": rec.psnr,
                    "ssim": rec.ssim,
                    "butteraugli": rec.butteraugli,
                    "chroma_subsampling": rec.chroma_subsampling,
                    "speed": rec.speed,
                    "effort": rec.effort,
                    "method": rec.method,
                    "resolution": rec.resolution,
                    "extra_args": rec.extra_args,
                    "measurement_error": rec.measurement_error,
                }
                for rec in self.measurements
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
