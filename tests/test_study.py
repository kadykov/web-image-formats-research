"""Tests for study configuration and parsing module."""

import json
from pathlib import Path

import pytest

from src.study import (
    EncoderConfig,
    StudyConfig,
    _parse_quality,
)


class TestParseQuality:
    """Tests for quality parameter parsing."""

    def test_single_integer(self) -> None:
        """Single integer produces a one-element list."""
        assert _parse_quality(75) == [75]

    def test_explicit_list(self) -> None:
        """Explicit list is returned as-is."""
        assert _parse_quality([60, 75, 90]) == [60, 75, 90]

    def test_range_object(self) -> None:
        """Range object is expanded to list."""
        result = _parse_quality({"start": 30, "stop": 60, "step": 10})
        assert result == [30, 40, 50, 60]

    def test_range_single_step(self) -> None:
        """Range with start == stop produces single element."""
        result = _parse_quality({"start": 50, "stop": 50, "step": 10})
        assert result == [50]

    def test_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid quality"):
            _parse_quality("bad")  # type: ignore[arg-type]


class TestStudyConfig:
    """Tests for StudyConfig loading and parsing."""

    def test_from_dict_minimal(self) -> None:
        """Minimal valid config."""
        data = {
            "id": "test-study",
            "dataset": {"id": "div2k-valid"},
            "encoders": [
                {"format": "webp", "quality": [75]},
            ],
        }
        config = StudyConfig.from_dict(data)
        assert config.id == "test-study"
        assert config.name == "test-study"  # defaults to id
        assert config.dataset_id == "div2k-valid"
        assert config.max_images is None
        assert len(config.encoders) == 1
        assert config.encoders[0].format == "webp"
        assert config.encoders[0].quality == [75]
        assert config.encoders[0].resolution is None

    def test_from_dict_full(self) -> None:
        """Full config with all optional fields."""
        data = {
            "id": "full-study",
            "name": "Full Study",
            "description": "Test study with all fields",
            "dataset": {"id": "div2k-valid", "max_images": 5},
            "encoders": [
                {
                    "format": "avif",
                    "quality": {"start": 30, "stop": 90, "step": 30},
                    "chroma_subsampling": ["444", "420"],
                    "speed": [4, 6],
                    "resolution": [1920, 1280],
                },
            ],
        }
        config = StudyConfig.from_dict(data)
        assert config.name == "Full Study"
        assert config.description == "Test study with all fields"
        assert config.max_images == 5
        enc = config.encoders[0]
        assert enc.format == "avif"
        assert enc.quality == [30, 60, 90]
        assert enc.chroma_subsampling == ["444", "420"]
        assert enc.speed == [4, 6]
        assert enc.resolution == [1920, 1280]

    def test_from_dict_missing_id_raises(self) -> None:
        """Missing required 'id' field raises ValueError."""
        with pytest.raises(ValueError, match="must have"):
            StudyConfig.from_dict({"dataset": {"id": "x"}, "encoders": []})

    def test_from_dict_missing_dataset_raises(self) -> None:
        """Missing 'dataset' field raises ValueError."""
        with pytest.raises(ValueError, match="must have"):
            StudyConfig.from_dict({"id": "x", "encoders": []})

    def test_from_dict_missing_encoders_raises(self) -> None:
        """Missing 'encoders' field raises ValueError."""
        with pytest.raises(ValueError, match="must have"):
            StudyConfig.from_dict({"id": "x", "dataset": {"id": "y"}})

    def test_from_file(self, tmp_path: Path) -> None:
        """Load config from a JSON file."""
        config_data = {
            "id": "file-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "jpeg", "quality": 75}],
        }
        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps(config_data))

        config = StudyConfig.from_file(config_file)
        assert config.id == "file-test"

    def test_from_file_not_found_raises(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            StudyConfig.from_file(tmp_path / "nonexistent.json")

    def test_encoder_with_single_speed(self) -> None:
        """Single speed integer is wrapped in a list."""
        data = {
            "id": "speed-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "avif", "quality": 75, "speed": 4}],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].speed == [4]

    def test_encoder_with_speed_list(self) -> None:
        """Speed list is preserved."""
        data = {
            "id": "speed-list-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "avif", "quality": 75, "speed": [2, 4, 6]}],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].speed == [2, 4, 6]

    def test_encoder_with_extra_args(self) -> None:
        """Extra args are preserved."""
        data = {
            "id": "extra-args-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [
                {
                    "format": "webp",
                    "quality": 75,
                    "extra_args": {"preset": "photo", "m": 6},
                }
            ],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].extra_args == {"preset": "photo", "m": 6}

    def test_time_budget(self) -> None:
        """Time budget is parsed from config."""
        data = {
            "id": "budget-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "jpeg", "quality": 75}],
            "time_budget": 3600,
        }
        config = StudyConfig.from_dict(data)
        assert config.time_budget == 3600

    def test_time_budget_default(self) -> None:
        """Time budget defaults to None."""
        data = {
            "id": "budget-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "jpeg", "quality": 75}],
        }
        config = StudyConfig.from_dict(data)
        assert config.time_budget is None


class TestEncoderConfigResolution:
    """Tests for resolution as a per-encoder sweep parameter."""

    def test_single_integer_resolution(self) -> None:
        """Single resolution integer is wrapped in a list."""
        data = {
            "id": "res-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "webp", "quality": 75, "resolution": 1280}],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].resolution == [1280]

    def test_resolution_list(self) -> None:
        """Resolution list is preserved."""
        data = {
            "id": "res-list-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [
                {"format": "webp", "quality": 75, "resolution": [2048, 1280, 720]},
            ],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].resolution == [2048, 1280, 720]

    def test_no_resolution(self) -> None:
        """No resolution means original images are used."""
        data = {
            "id": "no-res-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "webp", "quality": 75}],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].resolution is None

    def test_per_encoder_resolution(self) -> None:
        """Different encoders can have different resolutions."""
        data = {
            "id": "per-enc-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [
                {"format": "jpeg", "quality": 75, "resolution": [1280, 720]},
                {"format": "avif", "quality": 60, "resolution": [2048]},
                {"format": "webp", "quality": 80},
            ],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].resolution == [1280, 720]
        assert config.encoders[1].resolution == [2048]
        assert config.encoders[2].resolution is None

    def test_encoder_config_dataclass(self) -> None:
        """EncoderConfig can be constructed directly with resolution."""
        enc = EncoderConfig(
            format="avif",
            quality=[60, 80],
            resolution=[1920, 1280, 720],
            speed=[4, 6],
        )
        assert enc.resolution == [1920, 1280, 720]
        assert enc.speed == [4, 6]

    def test_effort_parsing(self) -> None:
        """Effort is parsed like speed."""
        data = {
            "id": "effort-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "jxl", "quality": 75, "effort": [3, 7]}],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].effort == [3, 7]

    def test_method_parsing(self) -> None:
        """Method is parsed like speed."""
        data = {
            "id": "method-test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "webp", "quality": 75, "method": [0, 4, 6]}],
        }
        config = StudyConfig.from_dict(data)
        assert config.encoders[0].method == [0, 4, 6]
