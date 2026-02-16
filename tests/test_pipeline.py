"""Tests for the merged encode+measure pipeline module."""

import json
from pathlib import Path

import pytest

from src.pipeline import (
    PipelineRunner,
    _build_output_name,
    _encode_and_measure,
    _error_record,
    _expand_encoder_tasks,
    _format_duration,
    _make_rel,
    parse_time_budget,
)
from src.quality import QualityRecord
from src.study import EncoderConfig, StudyConfig

# ---------------------------------------------------------------------------
# parse_time_budget
# ---------------------------------------------------------------------------


class TestParseTimeBudget:
    """Tests for time-budget string parsing."""

    def test_plain_seconds(self) -> None:
        assert parse_time_budget("3600") == 3600.0

    def test_float_seconds(self) -> None:
        assert parse_time_budget("90.5") == 90.5

    def test_hours(self) -> None:
        assert parse_time_budget("2h") == 7200.0

    def test_minutes(self) -> None:
        assert parse_time_budget("30m") == 1800.0

    def test_seconds_suffix(self) -> None:
        assert parse_time_budget("90s") == 90.0

    def test_combined_hm(self) -> None:
        assert parse_time_budget("1h30m") == 5400.0

    def test_combined_hms(self) -> None:
        assert parse_time_budget("2h15m30s") == 8130.0

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid time budget"):
            parse_time_budget("abc")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid time budget"):
            parse_time_budget("")


# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_seconds_only(self) -> None:
        assert _format_duration(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_duration(125) == "2m 05s"

    def test_hours(self) -> None:
        assert _format_duration(3661) == "1h 01m 01s"

    def test_negative(self) -> None:
        assert _format_duration(-1) == "—"

    def test_zero(self) -> None:
        assert _format_duration(0) == "0s"


# ---------------------------------------------------------------------------
# _make_rel
# ---------------------------------------------------------------------------


class TestMakeRel:
    """Tests for path relativisation."""

    def test_child_path(self) -> None:
        assert _make_rel(Path("/a/b/c.txt"), Path("/a")) == "b/c.txt"

    def test_unrelated_path(self) -> None:
        result = _make_rel(Path("/x/y/z.txt"), Path("/a"))
        assert result == "/x/y/z.txt"


# ---------------------------------------------------------------------------
# _build_output_name
# ---------------------------------------------------------------------------


class TestBuildOutputName:
    """Tests for output filename construction."""

    def test_basic(self) -> None:
        assert _build_output_name("img", 75) == "img_q75"

    def test_with_chroma(self) -> None:
        assert _build_output_name("img", 60, chroma_subsampling="420") == "img_q60_420"

    def test_with_speed(self) -> None:
        assert _build_output_name("img", 60, speed=4) == "img_q60_s4"

    def test_with_effort(self) -> None:
        assert _build_output_name("img", 60, effort=7) == "img_q60_e7"

    def test_with_method(self) -> None:
        assert _build_output_name("img", 60, method=6) == "img_q60_m6"

    def test_with_resolution(self) -> None:
        assert _build_output_name("img", 60, resolution=1280) == "img_q60_r1280"

    def test_all_params(self) -> None:
        name = _build_output_name("img", 60, chroma_subsampling="444", speed=4, resolution=1920)
        assert name == "img_q60_444_s4_r1920"


# ---------------------------------------------------------------------------
# _error_record
# ---------------------------------------------------------------------------


class TestErrorRecord:
    """Tests for error record creation."""

    def test_creates_record_with_error(self) -> None:
        rec = _error_record(
            source_image="src.png",
            original_image="orig.png",
            fmt="avif",
            quality=75,
            width=100,
            height=200,
            source_file_size=12345,
            chroma_subsampling="420",
            speed=4,
            effort=None,
            method=None,
            resolution=None,
            extra_args=None,
            error="encoding failed",
        )
        assert isinstance(rec, QualityRecord)
        assert rec.measurement_error == "encoding failed"
        assert rec.ssimulacra2 is None
        assert rec.format == "avif"
        assert rec.file_size == 0

    def test_with_file_size_and_encoding_time(self) -> None:
        rec = _error_record(
            source_image="s.png",
            original_image="o.png",
            fmt="jpeg",
            quality=85,
            width=0,
            height=0,
            source_file_size=0,
            chroma_subsampling=None,
            speed=None,
            effort=None,
            method=None,
            resolution=None,
            extra_args=None,
            error="measure failed",
            file_size=5000,
            encoding_time=1.5,
        )
        assert rec.file_size == 5000
        assert rec.encoding_time == 1.5


# ---------------------------------------------------------------------------
# _expand_encoder_tasks
# ---------------------------------------------------------------------------


class TestExpandEncoderTasks:
    """Tests for task expansion from encoder configs."""

    def test_basic_expansion(self, tmp_path: Path) -> None:
        """One encoder with 2 qualities = 2 tasks."""
        from PIL import Image

        img = tmp_path / "test.png"
        Image.new("RGB", (4, 4), "red").save(img)

        enc = EncoderConfig(format="webp", quality=[60, 80])
        tasks = _expand_encoder_tasks(
            source_image=img,
            original_image=img,
            enc=enc,
            resolution=None,
            save_artifact_dir=None,
            source_image_label=None,
        )
        assert len(tasks) == 2
        assert tasks[0]["quality"] == 60
        assert tasks[1]["quality"] == 80
        assert tasks[0]["save_dir_str"] is None

    def test_chroma_expansion(self, tmp_path: Path) -> None:
        """Chroma options multiply tasks."""
        from PIL import Image

        img = tmp_path / "test.png"
        Image.new("RGB", (4, 4), "red").save(img)

        enc = EncoderConfig(format="avif", quality=[75], chroma_subsampling=["444", "420"])
        tasks = _expand_encoder_tasks(
            source_image=img,
            original_image=img,
            enc=enc,
            resolution=None,
            save_artifact_dir=None,
            source_image_label=None,
        )
        assert len(tasks) == 2
        assert tasks[0]["chroma_subsampling"] == "444"
        assert tasks[1]["chroma_subsampling"] == "420"

    def test_save_dir(self, tmp_path: Path) -> None:
        """When save_artifact_dir is set, save_dir_str is populated."""
        from PIL import Image

        img = tmp_path / "test.png"
        Image.new("RGB", (4, 4), "red").save(img)

        enc = EncoderConfig(format="jpeg", quality=[85])
        save_dir = tmp_path / "encoded" / "study1"
        tasks = _expand_encoder_tasks(
            source_image=img,
            original_image=img,
            enc=enc,
            resolution=1280,
            save_artifact_dir=save_dir,
            source_image_label=None,
        )
        assert len(tasks) == 1
        assert "jpeg" in tasks[0]["save_dir_str"]
        assert "r1280" in tasks[0]["save_dir_str"]

    def test_full_combinatorial(self, tmp_path: Path) -> None:
        """Full combinatorial: quality × chroma × speed."""
        from PIL import Image

        img = tmp_path / "test.png"
        Image.new("RGB", (4, 4), "red").save(img)

        enc = EncoderConfig(
            format="avif",
            quality=[60, 80],
            chroma_subsampling=["444", "420"],
            speed=[4, 6],
        )
        tasks = _expand_encoder_tasks(
            source_image=img,
            original_image=img,
            enc=enc,
            resolution=None,
            save_artifact_dir=None,
            source_image_label=None,
        )
        # 2 quality × 2 chroma × 2 speed = 8
        assert len(tasks) == 8


# ---------------------------------------------------------------------------
# _encode_and_measure (integration-style, needs real encoder tools)
# ---------------------------------------------------------------------------


class TestEncodeAndMeasure:
    """Integration tests for the atomic encode+measure function.

    These tests require external tools (cjpeg, cwebp, etc.) to be
    installed. They are skipped if tools are not available.
    """

    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a small test image."""
        from PIL import Image

        img = tmp_path / "source.png"
        # Create a 64x64 image with some pattern for meaningful metrics
        image = Image.new("RGB", (64, 64))
        for x in range(64):
            for y in range(64):
                image.putpixel((x, y), (x * 4, y * 4, (x + y) * 2))
        image.save(img)
        return img

    @pytest.fixture
    def project_root(self, tmp_path: Path) -> Path:
        return tmp_path

    def _tool_available(self, tool: str) -> bool:
        import shutil

        return shutil.which(tool) is not None

    def test_jpeg_encode_and_measure(self, test_image: Path, project_root: Path) -> None:
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        record = _encode_and_measure(
            source_image=str(test_image),
            original_image=str(test_image),
            fmt="jpeg",
            quality=85,
            project_root_str=str(project_root),
        )
        assert isinstance(record, QualityRecord)
        assert record.format == "jpeg"
        assert record.quality == 85
        assert record.file_size > 0
        assert record.encoding_time is not None
        assert record.encoding_time > 0
        assert record.width == 64
        assert record.height == 64
        assert record.measurement_error is None
        # PSNR should be available (ffmpeg is typically present)
        # SSIMULACRA2 may or may not be available

    def test_webp_encode_and_measure(self, test_image: Path, project_root: Path) -> None:
        if not self._tool_available("cwebp"):
            pytest.skip("cwebp not available")

        record = _encode_and_measure(
            source_image=str(test_image),
            original_image=str(test_image),
            fmt="webp",
            quality=80,
            project_root_str=str(project_root),
            method=4,
        )
        assert record.format == "webp"
        assert record.file_size > 0
        assert record.measurement_error is None

    def test_unknown_format(self, test_image: Path, project_root: Path) -> None:
        record = _encode_and_measure(
            source_image=str(test_image),
            original_image=str(test_image),
            fmt="bmp",
            quality=50,
            project_root_str=str(project_root),
        )
        assert record.measurement_error is not None
        assert "Unknown format" in record.measurement_error

    def test_save_artifact(self, test_image: Path, project_root: Path, tmp_path: Path) -> None:
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        save_dir = tmp_path / "saved"
        record = _encode_and_measure(
            source_image=str(test_image),
            original_image=str(test_image),
            fmt="jpeg",
            quality=85,
            project_root_str=str(project_root),
            save_dir_str=str(save_dir),
        )
        assert record.encoded_path != ""
        assert save_dir.exists()
        saved_files = list(save_dir.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].suffix == ".jpg"

    def test_source_image_label(self, test_image: Path, project_root: Path) -> None:
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        record = _encode_and_measure(
            source_image=str(test_image),
            original_image=str(test_image),
            fmt="jpeg",
            quality=85,
            project_root_str=str(project_root),
            source_image_label="data/preprocessed/study/r1920/img_r1920.png",
        )
        assert record.source_image == "data/preprocessed/study/r1920/img_r1920.png"


# ---------------------------------------------------------------------------
# StudyConfig.time_budget
# ---------------------------------------------------------------------------


class TestStudyConfigTimeBudget:
    """Tests for time_budget in StudyConfig."""

    def test_from_dict_with_time_budget(self) -> None:
        data = {
            "id": "test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "jpeg", "quality": 75}],
            "time_budget": 3600,
        }
        config = StudyConfig.from_dict(data)
        assert config.time_budget == 3600

    def test_from_dict_without_time_budget(self) -> None:
        data = {
            "id": "test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "jpeg", "quality": 75}],
        }
        config = StudyConfig.from_dict(data)
        assert config.time_budget is None

    def test_from_file_with_time_budget(self, tmp_path: Path) -> None:
        data = {
            "id": "test",
            "dataset": {"id": "div2k-valid"},
            "encoders": [{"format": "jpeg", "quality": 85}],
            "time_budget": 1800,
        }
        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps(data))
        config = StudyConfig.from_file(config_file)
        assert config.time_budget == 1800


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    def test_collect_images(self, tmp_path: Path) -> None:
        """Collect images from a directory."""
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(5):
            Image.new("RGB", (4, 4), "red").save(img_dir / f"img{i:03d}.png")

        # Also create a non-image file
        (img_dir / "readme.txt").write_text("not an image")

        images = PipelineRunner._collect_images(img_dir)
        assert len(images) == 5

    def test_collect_images_max(self, tmp_path: Path) -> None:
        """Max images limits results."""
        from PIL import Image

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(10):
            Image.new("RGB", (4, 4), "red").save(img_dir / f"img{i:03d}.png")

        images = PipelineRunner._collect_images(img_dir, max_images=3)
        assert len(images) == 3

    def test_preprocess_image(self, tmp_path: Path) -> None:
        """Preprocessing resizes an image."""
        from PIL import Image

        source = tmp_path / "source.png"
        Image.new("RGB", (200, 100), "blue").save(source)

        out_dir = tmp_path / "prep"
        result = PipelineRunner._preprocess_image(source, 50, out_dir)

        assert result.exists()
        with Image.open(result) as img:
            # Longest edge should be ≤ 50
            assert max(img.size) <= 50


class TestPipelineRunnerIntegration:
    """Integration tests for PipelineRunner.run().

    These require a mock or real dataset and encoding tools.
    """

    @pytest.fixture
    def project_with_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal project structure with a tiny dataset."""
        from PIL import Image

        project = tmp_path / "project"
        project.mkdir()

        # Create dataset config
        config_dir = project / "config"
        config_dir.mkdir()
        datasets_json = config_dir / "datasets.json"
        datasets_json.write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "id": "test-ds",
                            "name": "Test Dataset",
                            "description": "Tiny test dataset",
                            "type": "zip",
                            "url": "http://example.com/test.zip",
                            "size_mb": 0.01,
                            "image_count": 3,
                            "resolution": "64px",
                            "format": "PNG",
                            "rename_to": "test_images",
                        }
                    ]
                }
            )
        )

        # Create dataset images
        ds_dir = project / "data" / "datasets" / "test_images"
        ds_dir.mkdir(parents=True)
        for i in range(5):
            img = Image.new("RGB", (64, 64))
            for x in range(64):
                for y in range(64):
                    img.putpixel(
                        (x, y), ((x * 4 + i * 20) % 256, (y * 4) % 256, ((x + y) * 2) % 256)
                    )
            img.save(ds_dir / f"img{i:03d}.png")

        return project

    def _tool_available(self, tool: str) -> bool:
        import shutil

        return shutil.which(tool) is not None

    def test_run_basic(self, project_with_dataset: Path) -> None:
        """Run pipeline with a simple JPEG config."""
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="test-study",
            name="Test Study",
            dataset_id="test-ds",
            max_images=2,
            encoders=[EncoderConfig(format="jpeg", quality=[75, 85])],
        )

        runner = PipelineRunner(project_with_dataset)
        results = runner.run(config)

        assert results.study_id == "test-study"
        assert results.dataset["image_count"] == 2
        # 2 images × 2 qualities = 4 measurements
        assert len(results.measurements) == 4
        for m in results.measurements:
            assert m.format == "jpeg"
            assert m.quality in (75, 85)
            assert m.file_size > 0

    def test_run_with_time_budget(self, project_with_dataset: Path) -> None:
        """Time budget limits number of images processed.

        With the worker-per-image model, all available workers are filled
        initially (up to num_workers images), then budget is checked before
        submitting additional images. This ensures no idle workers at startup.
        """
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="budget-test",
            name="Budget Test",
            dataset_id="test-ds",
            max_images=None,  # no max — budget controls
            encoders=[EncoderConfig(format="jpeg", quality=[75])],
        )

        runner = PipelineRunner(project_with_dataset)
        # With 0 second budget and 5 images available, all 5 images are submitted
        # to fill workers initially. Budget is only checked for additional submissions.
        results = runner.run(config, time_budget=0)

        # All 5 images should process (initial batch fills min(num_workers, num_images))
        assert results.dataset["image_count"] == 5

    def test_run_with_time_budget_completes_inflight(self, project_with_dataset: Path) -> None:
        """In-flight work completes after budget expires.

        With num_workers=2 and 5 images, initial batch submits 2 images.
        After first completes, another is submitted. After second completes,
        budget expires but the third in-flight image still completes.
        """
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="budget-inflight-test",
            name="Budget In-Flight Test",
            dataset_id="test-ds",
            max_images=None,
            encoders=[EncoderConfig(format="jpeg", quality=[75])],
        )

        runner = PipelineRunner(project_with_dataset)
        # Run with only 2 workers and tiny budget to trigger in-flight completion
        results = runner.run(config, time_budget=0.1, num_workers=2)

        # Should complete at least the initial 2 images, possibly more
        # if they finish quickly and can submit another before budget expires
        assert results.dataset["image_count"] >= 2
        # Should be <= 5 (total available)
        assert results.dataset["image_count"] <= 5
        # Verify measurements match image count
        assert len(results.measurements) == results.dataset["image_count"]

    def test_run_save_artifacts(self, project_with_dataset: Path) -> None:
        """Artifacts are saved when flag is set."""
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="save-test",
            name="Save Test",
            dataset_id="test-ds",
            max_images=1,
            encoders=[EncoderConfig(format="jpeg", quality=[85])],
        )

        runner = PipelineRunner(project_with_dataset)
        results = runner.run(config, save_artifacts=True)

        assert len(results.measurements) == 1
        rec = results.measurements[0]
        assert rec.encoded_path != ""

        # Check the file actually exists
        saved_path = project_with_dataset / rec.encoded_path
        assert saved_path.exists()

    def test_run_without_save_artifacts(self, project_with_dataset: Path) -> None:
        """Without save_artifacts, encoded_path is empty string."""
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="no-save-test",
            name="No Save Test",
            dataset_id="test-ds",
            max_images=1,
            encoders=[EncoderConfig(format="jpeg", quality=[85])],
        )

        runner = PipelineRunner(project_with_dataset)
        results = runner.run(config, save_artifacts=False)

        assert len(results.measurements) == 1
        assert results.measurements[0].encoded_path == ""

    def test_run_with_preprocessing(self, project_with_dataset: Path) -> None:
        """Pipeline with resize preprocessing."""
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="preprocess-test",
            name="Preprocess Test",
            dataset_id="test-ds",
            max_images=1,
            encoders=[EncoderConfig(format="jpeg", quality=[85])],
            resize=[32],  # downscale to 32px
        )

        runner = PipelineRunner(project_with_dataset)
        results = runner.run(config)

        assert len(results.measurements) == 1
        rec = results.measurements[0]
        # Image should be resized (32×32 or smaller)
        assert max(rec.width, rec.height) <= 32
        assert "preprocessed" in rec.source_image

    def test_run_multiple_resolutions(self, project_with_dataset: Path) -> None:
        """Multiple resolutions produce tasks for each."""
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="multi-res-test",
            name="Multi-Resolution Test",
            dataset_id="test-ds",
            max_images=1,
            encoders=[EncoderConfig(format="jpeg", quality=[85])],
            resize=[48, 32],  # two resolutions
        )

        runner = PipelineRunner(project_with_dataset)
        results = runner.run(config)

        # 1 image × 2 resolutions × 1 quality = 2 measurements
        assert len(results.measurements) == 2
        resolutions = {m.resolution for m in results.measurements}
        assert resolutions == {48, 32}

    def test_results_saveable(self, project_with_dataset: Path) -> None:
        """Results can be saved and loaded."""
        if not self._tool_available("cjpeg"):
            pytest.skip("cjpeg not available")

        config = StudyConfig(
            id="save-results-test",
            name="Save Results Test",
            dataset_id="test-ds",
            max_images=1,
            encoders=[EncoderConfig(format="jpeg", quality=[85])],
        )

        runner = PipelineRunner(project_with_dataset)
        results = runner.run(config)

        # Save
        out_path = project_with_dataset / "quality.json"
        results.save(out_path)
        assert out_path.exists()

        # Load and verify
        data = json.loads(out_path.read_text())
        assert data["study_id"] == "save-results-test"
        assert len(data["measurements"]) == 1
        assert data["measurements"][0]["format"] == "jpeg"

    def test_dataset_not_found(self, project_with_dataset: Path) -> None:
        """Missing dataset raises ValueError."""
        config = StudyConfig(
            id="missing-ds",
            name="Missing",
            dataset_id="nonexistent",
            max_images=None,
            encoders=[EncoderConfig(format="jpeg", quality=[75])],
        )
        runner = PipelineRunner(project_with_dataset)
        with pytest.raises(ValueError, match="not found"):
            runner.run(config)

    def test_dataset_not_downloaded(self, tmp_path: Path) -> None:
        """Dataset configured but not downloaded raises FileNotFoundError."""
        project = tmp_path / "project"
        project.mkdir()

        # Config exists but no dataset directory
        config_dir = project / "config"
        config_dir.mkdir()
        (config_dir / "datasets.json").write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "id": "not-fetched",
                            "name": "Not Fetched",
                            "description": "DS not fetched",
                            "type": "zip",
                            "url": "http://example.com/nf.zip",
                            "size_mb": 1,
                            "image_count": 10,
                            "resolution": "2K",
                            "format": "PNG",
                        }
                    ]
                }
            )
        )
        (project / "data" / "datasets").mkdir(parents=True)

        config = StudyConfig(
            id="nf-test",
            name="NF Test",
            dataset_id="not-fetched",
            max_images=None,
            encoders=[EncoderConfig(format="jpeg", quality=[75])],
        )
        runner = PipelineRunner(project)
        with pytest.raises(FileNotFoundError, match="Fetch it first"):
            runner.run(config)
