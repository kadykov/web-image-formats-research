"""Image preprocessing module.

This module handles image resizing and cropping for encoding studies.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class CropResult:
    """Result of a crop operation with region metadata.

    Attributes:
        path: Path to the cropped image file.
        crop_region: The region extracted from the original image as
            ``{"x": int, "y": int, "width": int, "height": int}``.
    """

    path: Path
    crop_region: dict[str, int]


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

    def crop_image_around_fragment(
        self,
        input_path: Path,
        fragment: dict[str, int],
        target_longest_edge: int,
        output_name: str | None = None,
        *,
        adjust_aspect_ratio: bool = False,
    ) -> CropResult:
        """Crop an image to a target size while keeping the analysis fragment.

        The crop window is sized so that its longest edge equals
        *target_longest_edge* (preserving the original aspect ratio),
        then positioned to center the analysis *fragment* as closely as
        possible.  The fragment is guaranteed to be fully contained in
        the crop window; the window is clamped to image boundaries.

        When *target_longest_edge* is greater than or equal to the
        original image's longest edge the original image is returned
        unchanged with a crop region covering the full image.

        Args:
            input_path: Path to the original image.
            fragment: Analysis fragment coordinates as
                ``{"x": int, "y": int, "width": int, "height": int}``
                in the original image coordinate space.
            target_longest_edge: Target longest edge of the crop window
                in pixels.
            output_name: Optional output filename.
            adjust_aspect_ratio: When ``True``, if the fragment is
                larger than the scaled crop dimensions the crop window
                is expanded (breaking aspect-ratio preservation) so
                that the fragment fits.  When ``False`` (default), a
                :class:`ValueError` is raised instead.

        Returns:
            :class:`CropResult` with the cropped image path and the
            crop region applied (in original image coordinates).

        Raises:
            ValueError: If the fragment cannot fit within the target
                crop dimensions and *adjust_aspect_ratio* is ``False``.
        """
        if output_name is None:
            output_name = f"{input_path.stem}_c{target_longest_edge}.png"
        output_path = self.output_dir / output_name

        with Image.open(input_path) as img:
            orig_w, orig_h = img.size
            orig_longest = max(orig_w, orig_h)

            # If target >= original longest edge, return the full image
            if target_longest_edge >= orig_longest:
                img.save(output_path)
                return CropResult(
                    path=output_path,
                    crop_region={"x": 0, "y": 0, "width": orig_w, "height": orig_h},
                )

            # Compute crop dimensions preserving aspect ratio
            scale = target_longest_edge / orig_longest
            crop_w = max(1, round(orig_w * scale))
            crop_h = max(1, round(orig_h * scale))

            fx, fy = fragment["x"], fragment["y"]
            fw, fh = fragment["width"], fragment["height"]

            if fw > crop_w or fh > crop_h:
                if adjust_aspect_ratio:
                    # Expand the crop to fit the fragment, breaking AR
                    crop_w = max(crop_w, fw)
                    crop_h = max(crop_h, fh)
                else:
                    msg = (
                        f"Analysis fragment ({fw}x{fh}) does not fit within "
                        f"target crop dimensions ({crop_w}x{crop_h}). "
                        f"Increase the minimum crop size or decrease "
                        f"analysis_fragment_size."
                    )
                    raise ValueError(msg)

            # Center the fragment within the crop window, then clamp
            frag_cx = fx + fw / 2
            frag_cy = fy + fh / 2

            crop_x = int(round(frag_cx - crop_w / 2))
            crop_y = int(round(frag_cy - crop_h / 2))

            # Clamp to image boundaries
            crop_x = max(0, min(crop_x, orig_w - crop_w))
            crop_y = max(0, min(crop_y, orig_h - crop_h))

            # Verify fragment is fully contained (should always hold
            # after clamping if fragment fits in crop dimensions)
            assert crop_x <= fx and crop_y <= fy
            assert crop_x + crop_w >= fx + fw and crop_y + crop_h >= fy + fh

            cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            cropped.save(output_path)

            return CropResult(
                path=output_path,
                crop_region={
                    "x": crop_x,
                    "y": crop_y,
                    "width": crop_w,
                    "height": crop_h,
                },
            )
