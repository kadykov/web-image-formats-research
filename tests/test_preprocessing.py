"""Tests for image preprocessing module."""

from pathlib import Path

from PIL import Image

from src.preprocessing import ImagePreprocessor
from tests.conftest import create_test_image


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


class TestCropImageAroundFragment:
    """Tests for crop_image_around_fragment."""

    def test_basic_crop(self, tmp_path: Path) -> None:
        """Crop reduces the image to the target longest edge."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "src.png", size=(400, 300))
        fragment = {"x": 150, "y": 100, "width": 100, "height": 100}

        result = preprocessor.crop_image_around_fragment(
            src, fragment=fragment, target_longest_edge=200
        )

        assert result.path.exists()
        with Image.open(result.path) as img:
            assert max(img.size) == 200
        assert result.crop_region is not None
        assert result.crop_region["width"] > 0
        assert result.crop_region["height"] > 0

    def test_crop_preserves_aspect_ratio(self, tmp_path: Path) -> None:
        """Cropped image preserves the original aspect ratio."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "src.png", size=(400, 200))
        fragment = {"x": 100, "y": 50, "width": 50, "height": 50}

        result = preprocessor.crop_image_around_fragment(
            src, fragment=fragment, target_longest_edge=200
        )

        with Image.open(result.path) as img:
            # 400x200 → 200x100 (2:1 ratio)
            assert img.size == (200, 100)

    def test_crop_contains_fragment(self, tmp_path: Path) -> None:
        """The crop region must contain the analysis fragment."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "src.png", size=(500, 500))
        fragment = {"x": 200, "y": 200, "width": 100, "height": 100}

        result = preprocessor.crop_image_around_fragment(
            src, fragment=fragment, target_longest_edge=300
        )

        cr = result.crop_region
        # Fragment must be within crop region
        assert cr["x"] <= fragment["x"]
        assert cr["y"] <= fragment["y"]
        assert cr["x"] + cr["width"] >= fragment["x"] + fragment["width"]
        assert cr["y"] + cr["height"] >= fragment["y"] + fragment["height"]

    def test_crop_larger_than_image_returns_original(self, tmp_path: Path) -> None:
        """When target crop is larger than image, return unchanged."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "src.png", size=(200, 100))
        fragment = {"x": 50, "y": 25, "width": 50, "height": 50}

        result = preprocessor.crop_image_around_fragment(
            src, fragment=fragment, target_longest_edge=300
        )

        # Should return original since 300 > 200 (longest edge)
        with Image.open(result.path) as img:
            assert img.size == (200, 100)

    def test_crop_at_image_edge(self, tmp_path: Path) -> None:
        """Fragment near image edge — crop must clamp to boundaries."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        src = create_test_image(tmp_path / "src.png", size=(400, 400))
        # Fragment at bottom-right corner
        fragment = {"x": 350, "y": 350, "width": 50, "height": 50}

        result = preprocessor.crop_image_around_fragment(
            src, fragment=fragment, target_longest_edge=200
        )

        cr = result.crop_region
        assert cr["x"] + cr["width"] <= 400
        assert cr["y"] + cr["height"] <= 400
        assert cr["x"] <= fragment["x"]
        assert cr["y"] <= fragment["y"]

    def test_fragment_too_large_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError when fragment doesn't fit in crop dimensions."""
        import pytest

        preprocessor = ImagePreprocessor(tmp_path / "output")
        # 400x200 image, crop to 100 longest edge → 100x50.
        # Fragment 80x80 doesn't fit in 50 (height).
        src = create_test_image(tmp_path / "src.png", size=(400, 200))
        fragment = {"x": 10, "y": 10, "width": 80, "height": 80}

        with pytest.raises(ValueError, match="does not fit"):
            preprocessor.crop_image_around_fragment(src, fragment=fragment, target_longest_edge=100)

    def test_adjust_aspect_ratio_expands_crop(self, tmp_path: Path) -> None:
        """adjust_aspect_ratio=True expands crop to fit fragment."""
        preprocessor = ImagePreprocessor(tmp_path / "output")
        # 400x200 image, crop to 100 longest edge → normally 100x50.
        # Fragment 80x80 doesn't fit height 50, so crop is expanded.
        src = create_test_image(tmp_path / "src.png", size=(400, 200))
        fragment = {"x": 10, "y": 10, "width": 80, "height": 80}

        result = preprocessor.crop_image_around_fragment(
            src,
            fragment=fragment,
            target_longest_edge=100,
            adjust_aspect_ratio=True,
        )

        with Image.open(result.path) as img:
            w, h = img.size
            # Width should be at least 80, height at least 80
            assert w >= 80
            assert h >= 80

        cr = result.crop_region
        # Fragment must be fully contained
        assert cr["x"] <= fragment["x"]
        assert cr["y"] <= fragment["y"]
        assert cr["x"] + cr["width"] >= fragment["x"] + fragment["width"]
        assert cr["y"] + cr["height"] >= fragment["y"] + fragment["height"]
