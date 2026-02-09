"""Tests for image encoding module."""

from pathlib import Path

from src.encoder import ImageEncoder


def test_encoder_initialization(tmp_path: Path) -> None:
    """Test ImageEncoder initialization."""
    encoder = ImageEncoder(tmp_path / "output")
    assert encoder.output_dir.exists()
