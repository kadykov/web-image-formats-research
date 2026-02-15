"""Integration tests for the measure_quality.py script."""

import json
import subprocess
from pathlib import Path

import pytest

from src.encoder import ImageEncoder
from src.study import StudyResults


@pytest.fixture
def test_image() -> Path:
    """Return path to the test fixture image."""
    return Path(__file__).parent / "fixtures" / "test_image.png"


@pytest.fixture
def test_image_ppm() -> Path:
    """Return path to the test fixture image in PPM format."""
    return Path(__file__).parent / "fixtures" / "test_image.ppm"


@pytest.fixture
def fake_encoding_results(tmp_path, test_image: Path, test_image_ppm: Path) -> Path:
    """Create a fake encoding results JSON for testing.

    This simulates the output of encode_images.py by encoding a test image
    in multiple formats and creating a results.json file.
    """
    # Create output directories
    encoded_dir = tmp_path / "encoded" / "test-study"
    encoded_dir.mkdir(parents=True)

    jpeg_dir = encoded_dir / "jpeg"
    webp_dir = encoded_dir / "webp"
    avif_dir = encoded_dir / "avif"
    jxl_dir = encoded_dir / "jxl"

    # Encode test image in multiple formats
    jpeg_encoder = ImageEncoder(jpeg_dir)
    jpeg_result = jpeg_encoder.encode_jpeg(test_image_ppm, quality=85)

    webp_encoder = ImageEncoder(webp_dir)
    webp_result = webp_encoder.encode_webp(test_image, quality=85)

    avif_encoder = ImageEncoder(avif_dir)
    avif_result = avif_encoder.encode_avif(test_image, quality=60, speed=6)

    jxl_encoder = ImageEncoder(jxl_dir)
    jxl_result = jxl_encoder.encode_jxl(test_image, quality=85)

    # Verify all encodings succeeded
    assert jpeg_result.success and jpeg_result.output_path
    assert webp_result.success and webp_result.output_path
    assert avif_result.success and avif_result.output_path
    assert jxl_result.success and jxl_result.output_path

    # Create a fake study results structure
    project_root = Path(__file__).parent.parent
    results = StudyResults(
        study_id="test-study",
        study_name="Test Study",
        dataset_id="test-dataset",
        dataset_path=str((Path(__file__).parent / "fixtures").relative_to(project_root)),
        image_count=1,
        timestamp="2026-02-11T10:00:00.000000+00:00",
        records=[],
    )

    # Add encoding records
    from src.study import EncodingRecord

    def make_relative(path: Path) -> str:
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return str(path)

    for result, fmt in [
        (jpeg_result, "jpeg"),
        (webp_result, "webp"),
        (avif_result, "avif"),
        (jxl_result, "jxl"),
    ]:
        record = EncodingRecord(
            source_image=make_relative(test_image),
            original_image=make_relative(test_image),
            encoded_path=make_relative(result.output_path),
            format=fmt,
            quality=85 if fmt != "avif" else 60,
            file_size=result.file_size or 0,
            width=100,
            height=100,
            source_file_size=test_image.stat().st_size,
            encoding_time=0.5,  # Mock encoding time in seconds
        )
        results.records.append(record)

    # Save results.json
    results_file = encoded_dir / "results.json"
    results.save(results_file)

    return results_file


def test_measure_quality_script_runs(fake_encoding_results: Path, tmp_path: Path):
    """Test that measure_quality.py script runs successfully."""
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "measure_quality.py"
    output_file = tmp_path / "quality.json"

    # Run the script
    result = subprocess.run(
        [
            "python3",
            str(script),
            str(fake_encoding_results),
            "--output",
            str(output_file),
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    # Check that script succeeded
    assert result.returncode == 0, (
        f"Script failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Check that output file was created
    assert output_file.exists(), "Output file not created"

    # Verify output JSON structure
    with open(output_file) as f:
        data = json.load(f)

    assert "study_id" in data
    assert "measurements" in data
    assert len(data["measurements"]) == 4  # JPEG, WebP, AVIF, JXL

    # Verify each measurement has quality metrics
    for measurement in data["measurements"]:
        assert "format" in measurement
        assert "quality" in measurement
        assert "ssimulacra2" in measurement
        assert "psnr" in measurement
        assert "ssim" in measurement
        assert "butteraugli" in measurement

        # At least PSNR and SSIM should be measured (always available via ffmpeg)
        assert measurement["psnr"] is not None, f"PSNR missing for {measurement['format']}"
        assert measurement["ssim"] is not None, f"SSIM missing for {measurement['format']}"


def test_measure_quality_script_with_workers(fake_encoding_results: Path, tmp_path: Path):
    """Test that measure_quality.py script works with custom worker count."""
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "measure_quality.py"
    output_file = tmp_path / "quality.json"

    # Run the script with 2 workers
    result = subprocess.run(
        [
            "python3",
            str(script),
            str(fake_encoding_results),
            "--output",
            str(output_file),
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_file.exists()


def test_measure_quality_script_handles_missing_file():
    """Test that measure_quality.py handles missing results file gracefully."""
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "measure_quality.py"

    result = subprocess.run(
        [
            "python3",
            str(script),
            "nonexistent.json",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    # Should exit with non-zero code
    assert result.returncode != 0
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


def test_measure_quality_script_help():
    """Test that measure_quality.py --help works."""
    project_root = Path(__file__).parent.parent
    script = project_root / "scripts" / "measure_quality.py"

    result = subprocess.run(
        ["python3", str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0
    assert "measure quality metrics" in result.stdout.lower()
    assert "--output" in result.stdout
    assert "--workers" in result.stdout
