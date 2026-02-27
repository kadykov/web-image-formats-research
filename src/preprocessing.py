"""Image preprocessing module.

This module handles image resizing for encoding at specific resolutions.
"""

from pathlib import Path

from PIL import Image


class ImagePreprocessor:
    """Handles image preprocessing operations."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the image preprocessor.

        Args:
            output_dir: Directory where preprocessed images will be stored
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resize_image(
        self,
        input_path: Path,
        target_size: tuple[int, int],
        output_name: str | None = None,
        keep_aspect_ratio: bool = True,
    ) -> Path:
        """Resize an image to target dimensions.

        Args:
            input_path: Path to the input image
            target_size: Target (width, height) in pixels
            output_name: Optional output filename (with extension)
            keep_aspect_ratio: If True, maintain aspect ratio and fit within target_size

        Returns:
            Path to the resized image
        """
        if output_name is None:
            output_name = f"{input_path.stem}_resized{input_path.suffix}"
        output_path = self.output_dir / output_name

        with Image.open(input_path) as img:
            resized_img = img
            if keep_aspect_ratio:
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
            else:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)  # type: ignore

            # Convert RGBA to RGB if saving as JPEG
            if output_path.suffix.lower() in [".jpg", ".jpeg"] and resized_img.mode == "RGBA":
                rgb_img = Image.new("RGB", resized_img.size, (255, 255, 255))
                rgb_img.paste(resized_img, mask=resized_img.split()[3])
                resized_img = rgb_img  # type: ignore

            resized_img.save(output_path)

        return output_path
