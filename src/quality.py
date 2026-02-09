"""Quality measurement module.

This module handles measuring image quality using various metrics
including SSIMULACRA2, PSNR, SSIM, and others.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path


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

    def measure_ssimulacra2(self, original: Path, compressed: Path) -> float | None:
        """Measure SSIMULACRA2 score between two images.

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            SSIMULACRA2 score (higher is better, 100 = lossless)
        """
        try:
            cmd = ["ssimulacra2", str(original), str(compressed)]
            result = subprocess.run(cmd, capture_output=True, check=True, text=True)
            # Parse the output to extract the score
            # The tool outputs just the score as a number
            output = result.stdout.strip()
            # Try to parse as float directly, or extract from formatted output
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

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            PSNR value in dB (higher is better)
        """
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(compressed),
                "-i",
                str(original),
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

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            SSIM value (0-1, higher is better)
        """
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(compressed),
                "-i",
                str(original),
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

        Args:
            original: Path to the original image
            compressed: Path to the compressed image

        Returns:
            Butteraugli distance (lower is better, <1.0 = excellent)
        """
        try:
            cmd = ["butteraugli_main", str(original), str(compressed)]
            result = subprocess.run(cmd, capture_output=True, check=True, text=True)
            # Parse the output to extract the distance
            # Format typically includes a single number or "distance = X.XX"
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
