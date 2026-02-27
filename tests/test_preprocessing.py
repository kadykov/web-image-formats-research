"""Tests for image preprocessing module."""

from pathlib import Path

from PIL import Image

from src.preprocessing import ImagePreprocessor


def create_test_image(
    path: Path,
    size: tuple[int, int] = (100, 80),
    mode: str = "RGB",
    color: tuple[int, ...] = (128, 128, 128),
) -> Path:
    """Create a test image at the given path."""
    img = Image.new(mode, size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    def test_resize_keep_aspect_ratio(self, tmp_path: Path) -> None:
        """Test resizing with aspect ratio preservation (thumbnail)."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "src.png", size=(200, 100))

        result = preprocessor.resize_image(src, (100, 100), keep_aspect_ratio=True)

        assert result.exists()
        with Image.open(result) as img:
            # Thumbnail preserves aspect ratio: 200x100 → 100x50
            assert img.size[0] <= 100
            assert img.size[1] <= 100

    def test_resize_no_aspect_ratio(self, tmp_path: Path) -> None:
        """Test resizing without aspect ratio preservation (exact resize)."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "src.png", size=(200, 100))

        result = preprocessor.resize_image(src, (50, 50), keep_aspect_ratio=False)

        assert result.exists()
        with Image.open(result) as img:
            assert img.size == (50, 50)

    def test_resize_rgba_to_jpeg(self, tmp_path: Path) -> None:
        """Test RGBA image saved as JPEG triggers RGB conversion."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        # Create an RGBA image
        src = tmp_path / "src.png"
        img = Image.new("RGBA", (100, 80), color=(128, 128, 128, 200))
        img.save(src)

        result = preprocessor.resize_image(
            src, (50, 50), output_name="out.jpg", keep_aspect_ratio=False
        )

        assert result.exists()
        with Image.open(result) as out_img:
            assert out_img.mode == "RGB"

    def test_resize_default_output_name(self, tmp_path: Path) -> None:
        """Test that default output name is generated from input."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "photo.png")

        result = preprocessor.resize_image(src, (50, 50))

        assert result.name == "photo_resized.png"
