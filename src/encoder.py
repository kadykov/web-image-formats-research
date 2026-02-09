"""Image encoding module.

This module handles encoding images to various formats (JPEG, WebP, AVIF, JPEG XL)
using external command-line tools.
"""

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

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}_q{quality}.jpg"

        try:
            cmd = ["cjpeg", "-quality", str(quality), str(input_path)]
            with open(output_path, "wb") as f:
                subprocess.run(cmd, stdout=f, check=True, stderr=subprocess.PIPE)

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

    def encode_webp(
        self, input_path: Path, quality: int, output_name: str | None = None
    ) -> EncodeResult:
        """Encode image to WebP format.

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}_q{quality}.webp"

        try:
            cmd = ["cwebp", "-q", str(quality), str(input_path), "-o", str(output_path)]
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
        output_name: str | None = None,
    ) -> EncodeResult:
        """Encode image to AVIF format.

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            speed: Encoding speed (0-10, higher is faster)
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}_q{quality}_s{speed}.avif"

        try:
            cmd = [
                "avifenc",
                "-s",
                str(speed),
                "-q",
                str(quality),
                str(input_path),
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

    def encode_jxl(
        self, input_path: Path, quality: int, output_name: str | None = None
    ) -> EncodeResult:
        """Encode image to JPEG XL format.

        Args:
            input_path: Path to the input image
            quality: Quality setting (0-100)
            output_name: Optional output filename (without extension)

        Returns:
            EncodeResult with encoding details
        """
        if output_name is None:
            output_name = input_path.stem
        output_path = self.output_dir / f"{output_name}_q{quality}.jxl"

        try:
            cmd = ["cjxl", str(input_path), str(output_path), "-q", str(quality)]
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
