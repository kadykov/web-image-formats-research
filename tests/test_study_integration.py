"""Integration tests for the study runner using real encoders.

These tests exercise the full StudyRunner pipeline against the small
test fixture images to verify end-to-end encoding and results writing.
"""

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.dataset import DatasetConfig
from src.study import StudyConfig, StudyRunner


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project layout with test images as the dataset."""
    # Copy test fixtures into a fake dataset directory
    fixtures = Path(__file__).parent / "fixtures"
    dataset_dir = tmp_path / "data" / "datasets" / "test-images"
    dataset_dir.mkdir(parents=True)
    shutil.copy(fixtures / "test_image.png", dataset_dir / "test_image.png")
    shutil.copy(fixtures / "test_image.ppm", dataset_dir / "test_image.ppm")

    return tmp_path


@pytest.fixture
def fake_dataset_config() -> DatasetConfig:
    """Return a DatasetConfig that matches the project_root fixture layout."""
    return DatasetConfig(
        id="test-images",
        name="Test Images",
        description="Tiny test images",
        type="local",
        url="",
        size_mb=0,
        image_count=2,
        resolution="16x16",
        format="PNG",
        rename_to="test-images",
    )


def _patch_dataset_lookup(config: DatasetConfig) -> Any:
    """Return a context manager that makes the runner find a fake dataset."""
    return patch(
        "src.study.DatasetFetcher.get_dataset_config",
        return_value=config,
    )


class TestStudyRunnerIntegration:
    """End-to-end tests for study execution with real encoders."""

    def test_jpeg_study(self, project_root: Path, fake_dataset_config: DatasetConfig) -> None:
        """Run a JPEG-only study and verify results."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-jpeg",
                "dataset": {"id": "test-images", "max_images": 1},
                "encoders": [{"format": "jpeg", "quality": [60, 80]}],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        assert results.study_id == "test-jpeg"
        assert results.dataset_id == "test-images"
        assert results.image_count == 1
        # 1 image × 2 quality levels = 2 records
        assert len(results.records) == 2
        for rec in results.records:
            assert rec.format == "jpeg"
            assert rec.file_size > 0
            assert rec.width > 0 and rec.height > 0
            encoded = project_root / rec.encoded_path
            assert encoded.exists()

    def test_webp_study(self, project_root: Path, fake_dataset_config: DatasetConfig) -> None:
        """Run a WebP-only study and verify results."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-webp",
                "dataset": {"id": "test-images", "max_images": 2},
                "encoders": [{"format": "webp", "quality": [75]}],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        # 2 images × 1 quality = 2 records
        assert len(results.records) == 2
        for rec in results.records:
            assert rec.format == "webp"
            assert rec.file_size > 0

    def test_avif_study(self, project_root: Path, fake_dataset_config: DatasetConfig) -> None:
        """Run an AVIF study with chroma subsampling variants."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-avif",
                "dataset": {"id": "test-images", "max_images": 1},
                "encoders": [
                    {
                        "format": "avif",
                        "quality": [60],
                        "chroma_subsampling": ["444", "420"],
                        "speed": 10,
                    }
                ],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        # 1 image × 1 quality × 2 chroma = 2 records
        assert len(results.records) == 2
        subsampling_values = {r.chroma_subsampling for r in results.records}
        assert subsampling_values == {"444", "420"}

    def test_jxl_study(self, project_root: Path, fake_dataset_config: DatasetConfig) -> None:
        """Run a JPEG XL study and verify results."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-jxl",
                "dataset": {"id": "test-images", "max_images": 1},
                "encoders": [{"format": "jxl", "quality": [75]}],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        assert len(results.records) == 1
        assert results.records[0].format == "jxl"
        assert results.records[0].file_size > 0

    def test_multi_format_study(
        self, project_root: Path, fake_dataset_config: DatasetConfig
    ) -> None:
        """Run a study with all four formats."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-multi",
                "dataset": {"id": "test-images", "max_images": 1},
                "encoders": [
                    {"format": "jpeg", "quality": [75]},
                    {"format": "webp", "quality": [75]},
                    {"format": "avif", "quality": [60], "speed": 10},
                    {"format": "jxl", "quality": [75]},
                ],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        # 1 image × 4 formats × 1 quality = 4 records
        assert len(results.records) == 4
        formats = {r.format for r in results.records}
        assert formats == {"jpeg", "webp", "avif", "jxl"}

    def test_results_save_roundtrip(
        self, project_root: Path, fake_dataset_config: DatasetConfig
    ) -> None:
        """Run a study, save results, and verify the JSON is well-formed."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-save",
                "dataset": {"id": "test-images", "max_images": 1},
                "encoders": [{"format": "webp", "quality": [75]}],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        output_path = project_root / "data" / "results" / "test-save.json"
        results.save(output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["study_id"] == "test-save"
        assert loaded["dataset"]["id"] == "test-images"
        assert len(loaded["encodings"]) == 1
        enc = loaded["encodings"][0]
        assert enc["format"] == "webp"
        assert enc["quality"] == 75
        assert enc["file_size"] > 0
        assert "source_image" in enc
        assert "encoded_path" in enc

    def test_output_filenames_are_descriptive(
        self, project_root: Path, fake_dataset_config: DatasetConfig
    ) -> None:
        """Verify that encoded filenames contain quality and other params."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-names",
                "dataset": {"id": "test-images", "max_images": 1},
                "encoders": [
                    {
                        "format": "avif",
                        "quality": [50],
                        "chroma_subsampling": ["420"],
                        "speed": 10,
                    }
                ],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        assert len(results.records) == 1
        encoded_name = Path(results.records[0].encoded_path).stem
        # Should contain quality and chroma info
        assert "q50" in encoded_name
        assert "420" in encoded_name

    def test_encoded_files_organized_by_format(
        self, project_root: Path, fake_dataset_config: DatasetConfig
    ) -> None:
        """Verify encoded files are organized into format/resolution subdirectories."""
        runner = StudyRunner(project_root)
        config = StudyConfig.from_dict(
            {
                "id": "test-dirs",
                "dataset": {"id": "test-images", "max_images": 1},
                "encoders": [
                    {"format": "jpeg", "quality": [75]},
                    {"format": "webp", "quality": [75]},
                ],
            }
        )

        with _patch_dataset_lookup(fake_dataset_config):
            results = runner.run(config)

        # Each format should live under data/encoded/<study_id>/<format>/original/
        for rec in results.records:
            encoded = project_root / rec.encoded_path
            assert encoded.exists()
            # Path should include format directory
            assert rec.format in str(rec.encoded_path)
