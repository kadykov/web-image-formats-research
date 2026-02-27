"""Shared test fixtures and helpers.

Provides common fixtures used across multiple test modules to eliminate
duplication. Each test module can still define its own specialised
fixtures when needed.
"""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tool_available(name: str) -> bool:
    """Check whether a CLI tool is available on PATH."""
    return shutil.which(name) is not None


def create_test_image(
    path: Path,
    size: tuple[int, int] = (64, 64),
    mode: str = "RGB",
    color: tuple[int, ...] = (128, 128, 128),
) -> Path:
    """Create a small test image and return its path."""
    img = Image.new(mode, size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Quality-data fixtures
# ---------------------------------------------------------------------------


def _make_measurement(
    *,
    source: str,
    fmt: str,
    quality: int,
    file_size: int,
    width: int = 1920,
    height: int = 1080,
    source_file_size: int = 1_000_000,
    ssimulacra2: float | None = 75.5,
    psnr: float | None = 38.5,
    ssim: float | None = 0.95,
    butteraugli: float | None = 2.5,
    chroma: str | None = None,
    speed: int | None = None,
    effort: int | None = None,
    method: int | None = None,
    resolution: int | None = None,
    error: str | None = None,
) -> dict:
    """Build a single measurement dict with sensible defaults."""
    return {
        "source_image": source,
        "original_image": source,
        "encoded_path": f"data/encoded/test/{fmt}/{Path(source).stem}_q{quality}.{fmt}",
        "format": fmt,
        "quality": quality,
        "file_size": file_size,
        "width": width,
        "height": height,
        "source_file_size": source_file_size,
        "ssimulacra2": ssimulacra2,
        "psnr": psnr,
        "ssim": ssim,
        "butteraugli": butteraugli,
        "chroma_subsampling": chroma,
        "speed": speed,
        "effort": effort,
        "method": method,
        "resolution": resolution,
        "extra_args": None,
        "measurement_error": error,
    }


@pytest.fixture
def sample_quality_data() -> dict:
    """Quality measurement data with 2 images × 2 quality levels (avif).

    Used by test_analysis, test_interactive, and any test that needs a
    simple single-format quality dataset.
    """
    return {
        "study_id": "test-study",
        "study_name": "Test Study",
        "dataset": {
            "id": "test-dataset",
            "path": "data/datasets/test",
            "image_count": 2,
        },
        "encoding_timestamp": "2026-02-11T12:00:00+00:00",
        "timestamp": "2026-02-11T12:30:00+00:00",
        "measurements": [
            _make_measurement(
                source="data/datasets/test/image1.png",
                fmt="avif",
                quality=50,
                file_size=100_000,
                ssimulacra2=75.5,
                psnr=38.5,
                ssim=0.95,
                butteraugli=2.5,
                chroma="420",
                speed=4,
            ),
            _make_measurement(
                source="data/datasets/test/image2.png",
                fmt="avif",
                quality=50,
                file_size=120_000,
                source_file_size=1_100_000,
                ssimulacra2=73.2,
                psnr=37.8,
                ssim=0.94,
                butteraugli=2.8,
                chroma="420",
                speed=4,
            ),
            _make_measurement(
                source="data/datasets/test/image1.png",
                fmt="avif",
                quality=70,
                file_size=150_000,
                ssimulacra2=82.3,
                psnr=40.2,
                ssim=0.97,
                butteraugli=1.8,
                chroma="420",
                speed=4,
            ),
            _make_measurement(
                source="data/datasets/test/image2.png",
                fmt="avif",
                quality=70,
                file_size=170_000,
                source_file_size=1_100_000,
                ssimulacra2=80.1,
                psnr=39.5,
                ssim=0.96,
                butteraugli=2.0,
                chroma="420",
                speed=4,
            ),
        ],
    }


@pytest.fixture
def multi_format_quality_data() -> dict:
    """Quality data with 3 formats × 2 qualities × 2 images."""
    measurements = []
    for fmt in ["jpeg", "webp", "avif"]:
        for q in [60, 80]:
            for img_idx in range(1, 3):
                measurements.append(
                    _make_measurement(
                        source=f"data/datasets/test/image{img_idx}.png",
                        fmt=fmt,
                        quality=q,
                        file_size=50_000 + q * 1000 + img_idx * 5000,
                        ssimulacra2=60.0 + q * 0.3 + img_idx,
                        psnr=30.0 + q * 0.1 + img_idx * 0.5,
                        ssim=0.85 + q * 0.001 + img_idx * 0.01,
                        butteraugli=4.0 - q * 0.03 + img_idx * 0.1,
                    )
                )

    return {
        "study_id": "multi-format-test",
        "study_name": "Multi-Format Test",
        "dataset": {
            "id": "test-dataset",
            "path": "data/datasets/test",
            "image_count": 2,
        },
        "encoding_timestamp": "2026-02-11T12:00:00+00:00",
        "timestamp": "2026-02-11T12:30:00+00:00",
        "measurements": measurements,
    }


@pytest.fixture
def quality_json_file(tmp_path: Path, sample_quality_data: dict) -> Path:
    """Write sample_quality_data to a JSON file and return its path."""
    path = tmp_path / "quality.json"
    with open(path, "w") as f:
        json.dump(sample_quality_data, f)
    return path
