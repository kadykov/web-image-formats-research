"""Tests for the quality measurement module."""

import json
import tempfile
from pathlib import Path

from src.quality import QualityRecord, QualityResults, WorstRegion


class TestWorstRegion:
    """Tests for WorstRegion dataclass."""

    def test_worst_region_creation(self):
        """Test creating a WorstRegion."""
        region = WorstRegion(x=10, y=20, width=128, height=128, avg_distortion=5.5)
        assert region.x == 10
        assert region.y == 20
        assert region.width == 128
        assert region.height == 128
        assert region.avg_distortion == 5.5


class TestQualityRecord:
    """Tests for QualityRecord dataclass."""

    def test_quality_record_creation(self):
        """Test creating a minimal QualityRecord."""
        record = QualityRecord(
            source_image="test.png",
            original_image="original.png",
            encoded_path="encoded.webp",
            format="webp",
            quality=80,
            file_size=1024,
            width=100,
            height=100,
            source_file_size=2048,
            ssimulacra2=85.5,
            psnr=40.0,
            ssim=0.95,
            butteraugli=0.5,
        )
        assert record.source_image == "test.png"
        assert record.quality == 80
        assert record.ssimulacra2 == 85.5
        assert record.measurement_error is None

    def test_quality_record_with_error(self):
        """Test creating a QualityRecord with an error."""
        record = QualityRecord(
            source_image="test.png",
            original_image="original.png",
            encoded_path="",
            format="webp",
            quality=80,
            file_size=0,
            width=100,
            height=100,
            source_file_size=2048,
            ssimulacra2=None,
            psnr=None,
            ssim=None,
            butteraugli=None,
            measurement_error="Encoding failed",
        )
        assert record.measurement_error == "Encoding failed"
        assert record.ssimulacra2 is None


class TestComputeSchemaPath:
    """Tests for QualityResults._compute_schema_path static method."""

    def test_standard_location_with_schema_found(self):
        """Test computing path when schema file exists (standard location)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema file
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Create output file in standard location
            output_dir = project_root / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            # Compute schema path
            result = QualityResults._compute_schema_path(output_file)

            # Should be exactly 3 levels up
            assert result == "../../../config/quality-results.schema.json"

    def test_fallback_when_schema_not_found(self):
        """Test fallback path when schema file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create output file without creating schema
            output_dir = Path(tmpdir) / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            # Should return fallback path
            result = QualityResults._compute_schema_path(output_file)

            # Fallback assumes standard location
            assert result == "../../../config/quality-results.schema.json"

    def test_nested_output_location(self):
        """Test computing path for deeply nested output location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Create output in deeply nested location
            output_dir = project_root / "data" / "metrics" / "study1" / "subdir"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            result = QualityResults._compute_schema_path(output_file)

            # Should account for extra depth
            assert result == "../../../../config/quality-results.schema.json"

    def test_forward_slashes_in_result(self):
        """Test that result uses forward slashes (JSON-compatible)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Create output file
            output_dir = project_root / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            result = QualityResults._compute_schema_path(output_file)

            # No backslashes even if on Windows
            assert "\\" not in result
            assert "/" in result

    def test_absolute_output_path(self):
        """Test that absolute output paths are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir).resolve()

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Create output file with absolute path
            output_dir = project_root / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = (output_dir / "quality.json").resolve()

            result = QualityResults._compute_schema_path(output_file)

            # Should still compute correct relative path
            assert result == "../../../config/quality-results.schema.json"


class TestQualityResultsSave:
    """Tests for QualityResults.save() method."""

    def test_save_creates_file(self, sample_quality_data: dict):
        """Test that save creates a JSON file."""
        # Convert sample quality data to QualityResults
        measurements = [QualityRecord(**m) for m in sample_quality_data["measurements"]]
        results = QualityResults(
            study_id=sample_quality_data["study_id"],
            study_name=sample_quality_data["study_name"],
            dataset=sample_quality_data["dataset"],
            measurements=measurements,
            timestamp=sample_quality_data["timestamp"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Save results
            output_dir = project_root / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            results.save(output_file)

            assert output_file.exists()
            assert output_file.stat().st_size > 0

    def test_save_includes_schema_field(self, sample_quality_data: dict):
        """Test that saved JSON includes $schema field."""
        # Convert sample quality data to QualityResults
        measurements = [QualityRecord(**m) for m in sample_quality_data["measurements"]]
        results = QualityResults(
            study_id=sample_quality_data["study_id"],
            study_name=sample_quality_data["study_name"],
            dataset=sample_quality_data["dataset"],
            measurements=measurements,
            timestamp=sample_quality_data["timestamp"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Save results
            output_dir = project_root / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            results.save(output_file)

            # Load and verify
            with open(output_file) as f:
                data = json.load(f)

            assert "$schema" in data
            assert data["$schema"] == "../../../config/quality-results.schema.json"

    def test_save_schema_is_first_field(self):
        """Test that $schema field appears first in JSON output."""
        results = QualityResults(
            study_id="test-study",
            study_name="Test Study",
            dataset={"id": "test-ds", "path": "data/datasets/test", "image_count": 1},
            measurements=[],
            timestamp="2026-03-01T12:00:00Z",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Save results
            output_dir = project_root / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            results.save(output_file)

            # Load and verify order
            with open(output_file) as f:
                data = json.load(f)

            # $schema should be the first key
            first_key = list(data.keys())[0]
            assert first_key == "$schema"

    def test_save_preserves_all_fields(self, sample_quality_data: dict):
        """Test that all result fields are preserved in JSON."""
        # Use sample data and add extra fields
        measurements = [QualityRecord(**m) for m in sample_quality_data["measurements"]]
        results = QualityResults(
            study_id=sample_quality_data["study_id"],
            study_name=sample_quality_data["study_name"],
            dataset=sample_quality_data["dataset"],
            measurements=measurements,
            timestamp=sample_quality_data["timestamp"],
            encoding_timestamp=sample_quality_data.get("encoding_timestamp"),
            tool_versions={"cjpeg": "2.1.5", "cwebp": "1.5.0"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Save results
            output_dir = project_root / "data" / "metrics" / "test-study"
            output_dir.mkdir(parents=True)
            output_file = output_dir / "quality.json"

            results.save(output_file)

            # Load and verify all fields
            with open(output_file) as f:
                data = json.load(f)

            assert data["study_id"] == "test-study"
            assert data["study_name"] == "Test Study"
            assert data["encoding_timestamp"] == "2026-02-11T12:00:00+00:00"
            assert data["timestamp"] == "2026-02-11T12:30:00+00:00"
            assert data["tool_versions"]["cjpeg"] == "2.1.5"
            assert len(data["measurements"]) == 4
            assert data["measurements"][0]["ssimulacra2"] == 75.5
            assert data["measurements"][0]["chroma_subsampling"] == "420"

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories if they don't exist."""
        results = QualityResults(
            study_id="test-study",
            study_name="Test Study",
            dataset={"id": "test-ds", "path": "data/datasets/test", "image_count": 1},
            measurements=[],
            timestamp="2026-03-01T12:00:00Z",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create config directory with schema
            config_dir = project_root / "config"
            config_dir.mkdir()
            schema_file = config_dir / "quality-results.schema.json"
            schema_file.write_text("{}")

            # Save to path with non-existent parent directories
            output_file = project_root / "data" / "metrics" / "test-study" / "quality.json"

            # Parent directories should not exist yet
            assert not output_file.parent.exists()

            results.save(output_file)

            # Parent directories should be created
            assert output_file.exists()
            assert output_file.parent.exists()
