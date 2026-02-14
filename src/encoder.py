"""Image encoding module.

This module handles encoding images to various formats (JPEG, WebP, AVIF, JPEG XL)
using external command-line tools.

All encode methods force single-threaded mode so that parallelism is
handled at the task level (one process per encoding task).
"""

import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EncodeResult:
    """Result of an encoding operation."""

    success: bool
    output_path: Path | None
    file_size: int | None
    error_message: str | None = None


def get_encoder_version(encoder: str) -> str | None:
    """Get version string for an encoder tool.

    Args:
        encoder: Name of encoder tool (cjpeg, cwebp, avifenc, cjxl)

    Returns:
        Version string or None if unable to determine
    """
    try:
        if encoder == "cjpeg":
            # libjpeg-turbo: "cjpeg: JPEG file conversion from PPM/PGM/BMP/Targa files"
            # Usually shows version with -version but not all builds support it
            result = subprocess.run(
                ["cjpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            # Extract version like "2.1.5"
            match = re.search(r"version\s+(\d+\.\d+\.\d+)", output, re.IGNORECASE)
            if match:
                return match.group(1)
            # Fallback: extract libjpeg-turbo version
            match = re.search(r"libjpeg-turbo\s+(\d+\.\d+\.\d+)", output, re.IGNORECASE)
            if match:
                return f"libjpeg-turbo {match.group(1)}"
            return "unknown"

        elif encoder == "cwebp":
            # cwebp: "1.5.0" (simple version output)
            result = subprocess.run(
                ["cwebp", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            # Try to extract version from first line
            lines = output.strip().split("\n")
            if lines:
                match = re.search(r"(\d+\.\d+\.\d+)", lines[0])
                if match:
                    return match.group(1)
            return "unknown"

        elif encoder == "avifenc":
            # avifenc: "avifenc version: 1.0.3"
            result = subprocess.run(
                ["avifenc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            match = re.search(r"version:\s*(\d+\.\d+\.\d+)", output, re.IGNORECASE)
            if match:
                return match.group(1)
            return "unknown"

        elif encoder == "cjxl":
            # cjxl: "JPEG XL encoder v0.10.2"
            result = subprocess.run(
                ["cjxl", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            match = re.search(r"v?(\d+\.\d+\.\d+)", output)
            if match:
                return match.group(1)
            return "unknown"

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


class ImageEncoder:
    """Handles encoding images to various formats."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the image encoder.

        Args:
            output_dir: Directory where encoded images will be stored
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def encode_jpeg(
        self, input_path: Path, quality: int, output_name: str | None = None
    ) -> EncodeResult:
        """Encode image to JPEG format.

        For inputs that ``cjpeg`` cannot read directly (e.g. PNG), the image
        is converted to PPM in memory and piped to ``cjpeg`` via *stdin*,
        avoiding any temporary files on disk.

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}.jpg"

        try:
            cjpeg_native_formats = {".ppm", ".pgm", ".bmp", ".tga"}
            if input_path.suffix.lower() in cjpeg_native_formats:
                # cjpeg can read this format directly from a file
                cmd = ["cjpeg", "-quality", str(quality), str(input_path)]
                with open(output_path, "wb") as f:
                    subprocess.run(cmd, stdout=f, check=True, stderr=subprocess.PIPE)
            else:
                # Convert to PPM in memory and pipe to cjpeg via stdin
                ppm_data = self._to_ppm_bytes(input_path)
                cmd = ["cjpeg", "-quality", str(quality)]
                with open(output_path, "wb") as f:
                    subprocess.run(
                        cmd, input=ppm_data, stdout=f, check=True, stderr=subprocess.PIPE
                    )

            return EncodeResult(
                success=True, output_path=output_path, file_size=output_path.stat().st_size
            )
        except subprocess.CalledProcessError as e:
            return EncodeResult(
                success=False,
                output_path=None,
                file_size=None,
                error_message=e.stderr.decode() if e.stderr else str(e),
            )

    @staticmethod
    def _to_ppm_bytes(input_path: Path) -> bytes:
        """Convert an image to PPM bytes in memory.

        Args:
            input_path: Path to the source image

        Returns:
            Raw PPM data suitable for piping to cjpeg
        """
        from PIL import Image

        buf = io.BytesIO()
        with Image.open(input_path) as img:
            img.save(buf, format="PPM")
        return buf.getvalue()

    def encode_webp(
        self,
        input_path: Path,
        quality: int,
        method: int = 4,
        output_name: str | None = None,
    ) -> EncodeResult:
        """Encode image to WebP format.

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            method: Compression method (0=fast, 6=slowest), default=4
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}.webp"

        try:
            cmd = [
                "cwebp",
                "-q",
                str(quality),
                "-m",
                str(method),
                str(input_path),
                "-o",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            return EncodeResult(
                success=True, output_path=output_path, file_size=output_path.stat().st_size
            )
        except subprocess.CalledProcessError as e:
            return EncodeResult(
                success=False,
                output_path=None,
                file_size=None,
                error_message=e.stderr.decode() if e.stderr else str(e),
            )

    def encode_avif(
        self,
        input_path: Path,
        quality: int,
        speed: int = 4,
        chroma_subsampling: str | None = None,
        output_name: str | None = None,
    ) -> EncodeResult:
        """Encode image to AVIF format.

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            speed: Encoding speed (0-10, higher is faster)
            chroma_subsampling: Chroma subsampling mode ("444", "422", "420", "400").
                If None, avifenc default is used.
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}.avif"

        try:
            cmd = [
                "avifenc",
                "-j",
                "1",
                "-s",
                str(speed),
                "-q",
                str(quality),
            ]
            if chroma_subsampling is not None:
                cmd.extend(["-y", chroma_subsampling])
            cmd.extend([str(input_path), str(output_path)])
            subprocess.run(cmd, check=True, capture_output=True)

            return EncodeResult(
                success=True, output_path=output_path, file_size=output_path.stat().st_size
            )
        except subprocess.CalledProcessError as e:
            return EncodeResult(
                success=False,
                output_path=None,
                file_size=None,
                error_message=e.stderr.decode() if e.stderr else str(e),
            )

    def encode_jxl(
        self,
        input_path: Path,
        quality: int,
        effort: int = 7,
        output_name: str | None = None,
    ) -> EncodeResult:
        """Encode image to JPEG XL format.

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            effort: Encoder effort setting (1-10, higher is slower/better), default=7
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}.jxl"

        try:
            cmd = [
                "cjxl",
                str(input_path),
                str(output_path),
                "-q",
                str(quality),
                "-e",
                str(effort),
                "--num_threads=1",
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            return EncodeResult(
                success=True, output_path=output_path, file_size=output_path.stat().st_size
            )
        except subprocess.CalledProcessError as e:
            return EncodeResult(
                success=False,
                output_path=None,
                file_size=None,
                error_message=e.stderr.decode() if e.stderr else str(e),
            )
