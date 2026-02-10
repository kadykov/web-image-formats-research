"""Tests for study configuration and execution module."""

import json
from pathlib import Path

import pytest

from src.study import (
    EncoderConfig,
    EncodingTask,
    StudyConfig,
    StudyResults,
    _build_output_name,
    _interleave_tasks,
    _parse_quality,
    expand_tasks,
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
        assert config.resize is None
        assert len(config.encoders) == 1
        assert config.encoders[0].format == "webp"
        assert config.encoders[0].quality == [75]

    def test_from_dict_full(self) -> None:
        """Full config with all optional fields."""
        data = {
            "id": "full-study",
            "name": "Full Study",
            "description": "Test study with all fields",
            "dataset": {"id": "div2k-valid", "max_images": 5},
            "preprocessing": {"resize": [1920, 1280]},
            "encoders": [
                {
                    "format": "avif",
                    "quality": {"start": 30, "stop": 90, "step": 30},
                    "chroma_subsampling": ["444", "420"],
                    "speed": [4, 6],
                },
            ],
        }
        config = StudyConfig.from_dict(data)
        assert config.name == "Full Study"
        assert config.description == "Test study with all fields"
        assert config.max_images == 5
        assert config.resize == [1920, 1280]
        enc = config.encoders[0]
        assert enc.format == "avif"
        assert enc.quality == [30, 60, 90]
        assert enc.chroma_subsampling == ["444", "420"]
        assert enc.speed == [4, 6]

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


class TestExpandTasks:
    """Tests for task expansion from study configs."""

    @pytest.fixture
    def images(self, tmp_path: Path) -> list[Path]:
        """Create two tiny real PNG files for expand_tasks."""
        from PIL import Image

        paths: list[Path] = []
        for name in ("a.png", "b.png"):
            p = tmp_path / name
            Image.new("RGB", (4, 4), "red").save(p)
            paths.append(p)
        return paths

    def _make_config(self, **kwargs: object) -> StudyConfig:
        """Helper to create a minimal config with overrides."""
        defaults: dict = {
            "id": "test",
            "name": "Test",
            "dataset_id": "div2k-valid",
            "max_images": None,
            "encoders": [
                EncoderConfig(format="webp", quality=[60, 80]),
            ],
        }
        defaults.update(kwargs)
        return StudyConfig(**defaults)

    def test_basic_expansion(self, images: list[Path]) -> None:
        """Tasks are expanded for each image × quality."""
        config = self._make_config()
        tasks = expand_tasks(config, images)
        # 2 images × 2 quality levels = 4 tasks
        assert len(tasks) == 4
        assert all(isinstance(t, EncodingTask) for t in tasks)
        # megapixels should be computed (4×4 = 16px = 0.000016 MP)
        assert all(t.megapixels > 0 for t in tasks)

    def test_chroma_subsampling_expansion(self, images: list[Path]) -> None:
        """Tasks multiply for each chroma subsampling mode."""
        config = self._make_config(
            encoders=[
                EncoderConfig(
                    format="avif",
                    quality=[75],
                    chroma_subsampling=["444", "420"],
                ),
            ]
        )
        tasks = expand_tasks(config, images[:1])
        # 1 image × 1 quality × 2 subsampling = 2 tasks
        assert len(tasks) == 2
        assert tasks[0].chroma_subsampling == "444"
        assert tasks[1].chroma_subsampling == "420"

    def test_speed_expansion(self, images: list[Path]) -> None:
        """Tasks multiply for each speed setting."""
        config = self._make_config(
            encoders=[
                EncoderConfig(format="avif", quality=[75], speed=[4, 6]),
            ]
        )
        tasks = expand_tasks(config, images[:1])
        # 1 image × 1 quality × 2 speeds = 2 tasks
        assert len(tasks) == 2
        assert tasks[0].speed == 4
        assert tasks[1].speed == 6

    def test_full_combinatorial(self, images: list[Path]) -> None:
        """Full combinatorial expansion: images × quality × chroma × speed."""
        config = self._make_config(
            encoders=[
                EncoderConfig(
                    format="avif",
                    quality=[60, 80],
                    chroma_subsampling=["444", "420"],
                    speed=[4, 6],
                ),
            ]
        )
        tasks = expand_tasks(config, images)
        # 2 images × 2 quality × 2 chroma × 2 speed = 16 tasks
        assert len(tasks) == 16

    def test_multiple_encoders(self, images: list[Path]) -> None:
        """Tasks from multiple encoders are combined."""
        config = self._make_config(
            encoders=[
                EncoderConfig(format="jpeg", quality=[75]),
                EncoderConfig(format="webp", quality=[75]),
            ]
        )
        tasks = expand_tasks(config, images[:1])
        assert len(tasks) == 2
        assert tasks[0].format == "jpeg"
        assert tasks[1].format == "webp"

    def test_resolution_tag(self, images: list[Path]) -> None:
        """Resolution tag is attached to all tasks."""
        config = self._make_config(encoders=[EncoderConfig(format="webp", quality=[75])])
        tasks = expand_tasks(config, images[:1], resolution=1280)
        assert len(tasks) == 1
        assert tasks[0].resolution == 1280

    def test_original_images_tracked(self, images: list[Path], tmp_path: Path) -> None:
        """Original images are tracked separately from source images."""
        from PIL import Image

        resized = tmp_path / "a_resized.png"
        Image.new("RGB", (2, 2), "blue").save(resized)
        source = [resized]
        original = [images[0]]
        config = self._make_config(encoders=[EncoderConfig(format="webp", quality=[75])])
        tasks = expand_tasks(config, source, original_images=original)
        assert tasks[0].source_image == source[0]
        assert tasks[0].original_image == original[0]


class TestBuildOutputName:
    """Tests for output filename generation."""

    def test_basic_name(self) -> None:
        """Basic name with just quality."""
        task = EncodingTask(
            source_image=Path("x.png"),
            original_image=Path("x.png"),
            format="webp",
            quality=75,
            megapixels=0.0,
        )
        name = _build_output_name(task, "image001")
        assert name == "image001_q75"

    def test_with_chroma(self) -> None:
        """Name includes chroma subsampling."""
        task = EncodingTask(
            source_image=Path("x.png"),
            original_image=Path("x.png"),
            format="avif",
            quality=60,
            megapixels=0.0,
            chroma_subsampling="420",
        )
        name = _build_output_name(task, "image001")
        assert name == "image001_q60_420"

    def test_with_speed(self) -> None:
        """Name includes speed."""
        task = EncodingTask(
            source_image=Path("x.png"),
            original_image=Path("x.png"),
            format="avif",
            quality=60,
            megapixels=0.0,
            speed=4,
        )
        name = _build_output_name(task, "image001")
        assert name == "image001_q60_s4"

    def test_with_resolution(self) -> None:
        """Name includes resolution."""
        task = EncodingTask(
            source_image=Path("x.png"),
            original_image=Path("x.png"),
            format="avif",
            quality=60,
            megapixels=0.0,
            resolution=1280,
        )
        name = _build_output_name(task, "image001")
        assert name == "image001_q60_r1280"

    def test_all_parameters(self) -> None:
        """Name includes all parameters."""
        task = EncodingTask(
            source_image=Path("x.png"),
            original_image=Path("x.png"),
            format="avif",
            quality=60,
            megapixels=0.0,
            chroma_subsampling="444",
            speed=4,
            resolution=1920,
        )
        name = _build_output_name(task, "img")
        assert name == "img_q60_444_s4_r1920"


class TestStudyResults:
    """Tests for StudyResults serialization."""

    def test_to_dict(self) -> None:
        """Results serialize to the expected schema shape."""
        results = StudyResults(
            study_id="test",
            study_name="Test Study",
            dataset_id="div2k-valid",
            dataset_path="data/datasets/DIV2K_valid",
            image_count=10,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        d = results.to_dict()
        assert d["study_id"] == "test"
        assert d["study_name"] == "Test Study"
        assert d["dataset"]["id"] == "div2k-valid"
        assert d["dataset"]["path"] == "data/datasets/DIV2K_valid"
        assert d["dataset"]["image_count"] == 10
        assert d["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert d["encodings"] == []

    def test_save(self, tmp_path: Path) -> None:
        """Results are written to JSON file."""
        results = StudyResults(
            study_id="test",
            study_name="Test",
            dataset_id="d",
            dataset_path="data/datasets/d",
            image_count=1,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        output_path = tmp_path / "out" / "results.json"
        results.save(output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["study_id"] == "test"
        assert loaded["encodings"] == []


class TestStudyConfigFromExampleFiles:
    """Tests that validate the example study files in studies/ directory."""

    @pytest.fixture
    def studies_dir(self) -> Path:
        """Return the studies directory."""
        return Path(__file__).parent.parent / "config" / "studies"

    def test_avif_quality_sweep(self, studies_dir: Path) -> None:
        """Validate avif-quality-sweep.json."""
        config = StudyConfig.from_file(studies_dir / "avif-quality-sweep.json")
        assert config.id == "avif-quality-sweep"
        assert config.dataset_id == "div2k-valid"
        assert len(config.encoders) == 1
        assert config.encoders[0].format == "avif"
        assert config.encoders[0].chroma_subsampling == ["444", "420"]
        # Quality range: 30 to 90 step 5 → 13 values
        assert len(config.encoders[0].quality) == 13

    def test_format_comparison(self, studies_dir: Path) -> None:
        """Validate format-comparison.json."""
        config = StudyConfig.from_file(studies_dir / "format-comparison.json")
        assert config.id == "format-comparison"
        assert len(config.encoders) == 4
        formats = [e.format for e in config.encoders]
        assert set(formats) == {"jpeg", "webp", "avif", "jxl"}

    def test_resolution_impact(self, studies_dir: Path) -> None:
        """Validate resolution-impact.json."""
        config = StudyConfig.from_file(studies_dir / "resolution-impact.json")
        assert config.id == "resolution-impact"
        assert config.resize == [2048, 1920, 1280, 960, 640]
        assert len(config.encoders) == 1


class TestInterleaveTasks:
    """Tests for task interleaving across formats."""

    @staticmethod
    def _task(fmt: str, quality: int = 75) -> EncodingTask:
        """Create a minimal EncodingTask for testing interleaving."""
        return EncodingTask(
            source_image=Path("x.png"),
            original_image=Path("x.png"),
            format=fmt,
            quality=quality,
            megapixels=1.0,
        )

    def test_single_format_unchanged(self) -> None:
        """Single format list is returned in original order."""
        tasks = [self._task("avif", q) for q in (30, 60, 90)]
        result = _interleave_tasks(tasks)
        assert [t.quality for t in result] == [30, 60, 90]

    def test_two_formats_alternate(self) -> None:
        """Two formats with equal counts alternate perfectly."""
        tasks = [
            self._task("jpeg", 60),
            self._task("jpeg", 80),
            self._task("avif", 60),
            self._task("avif", 80),
        ]
        result = _interleave_tasks(tasks)
        formats = [t.format for t in result]
        assert formats == ["jpeg", "avif", "jpeg", "avif"]

    def test_unequal_counts(self) -> None:
        """Longer bucket's tail is appended after shorter runs out."""
        tasks = [
            self._task("jpeg", 75),
            self._task("avif", 30),
            self._task("avif", 60),
            self._task("avif", 90),
        ]
        result = _interleave_tasks(tasks)
        formats = [t.format for t in result]
        assert formats == ["jpeg", "avif", "avif", "avif"]

    def test_empty_input(self) -> None:
        """Empty list returns empty list."""
        assert _interleave_tasks([]) == []

    def test_preserves_all_tasks(self) -> None:
        """Interleaving does not lose or duplicate tasks."""
        tasks = [self._task(f, q) for f in ("jpeg", "webp", "avif") for q in (60, 80)]
        result = _interleave_tasks(tasks)
        assert len(result) == 6
        assert {id(t) for t in result} == {id(t) for t in tasks}
