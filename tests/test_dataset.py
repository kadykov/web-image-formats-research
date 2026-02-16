"""Tests for dataset fetching module."""

import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.dataset import DatasetConfig, DatasetFetcher


def create_test_config(tmp_path: Path) -> Path:
    """Create a test datasets.json configuration file.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to the created config file
    """
    config = {
        "datasets": [
            {
                "id": "test-dataset",
                "name": "Test Dataset",
                "description": "A test dataset for unit tests",
                "type": "zip",
                "url": "http://example.com/test.zip",
                "size_mb": 10,
                "image_count": 5,
                "resolution": "1080p",
                "format": "PNG",
                "extracted_folder": "test_extracted",
                "rename_to": "test_dataset",
            },
            {
                "id": "div2k-valid",
                "name": "DIV2K Validation",
                "description": "DIV2K validation set",
                "type": "zip",
                "url": "http://example.com/div2k_valid.zip",
                "size_mb": 500,
                "image_count": 100,
                "resolution": "2K",
                "format": "PNG",
                "extracted_folder": "DIV2K_valid_HR",
                "rename_to": "DIV2K_valid",
            },
        ]
    }
    config_file = tmp_path / "test_datasets.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    return config_file


def test_dataset_config_from_dict() -> None:
    """Test DatasetConfig creation from dictionary."""
    data = {
        "id": "test-id",
        "name": "Test Name",
        "description": "Test Description",
        "type": "zip",
        "url": "http://example.com/test.zip",
        "size_mb": 100,
        "image_count": 50,
        "resolution": "2K",
        "format": "PNG",
    }
    config = DatasetConfig.from_dict(data)
    assert config.id == "test-id"
    assert config.name == "Test Name"
    assert config.size_mb == 100


def test_dataset_config_from_dict_with_cloud_storage() -> None:
    """Test DatasetConfig creation with cloud storage fields."""
    data = {
        "id": "test-gdrive",
        "name": "Test Google Drive",
        "description": "Test dataset on Google Drive",
        "type": "folder",
        "url": "https://drive.google.com/drive/folders/test123",
        "size_mb": 500,
        "image_count": 200,
        "resolution": "4K",
        "format": "PNG",
        "storage_type": "google_drive",
        "folder_id": "test123",
    }
    config = DatasetConfig.from_dict(data)
    assert config.id == "test-gdrive"
    assert config.storage_type == "google_drive"
    assert config.folder_id == "test123"
    assert config.type == "folder"


def test_dataset_config_defaults() -> None:
    """Test DatasetConfig defaults for optional fields."""
    data = {
        "id": "minimal",
        "name": "Minimal",
        "description": "Minimal config",
        "type": "zip",
        "url": "http://example.com/test.zip",
        "size_mb": 10,
        "image_count": 5,
        "resolution": "1K",
        "format": "JPEG",
    }
    config = DatasetConfig.from_dict(data)
    assert config.storage_type == "direct"  # default value
    assert config.folder_id is None
    assert config.extracted_folder is None


def test_dataset_fetcher_initialization(tmp_path: Path) -> None:
    """Test DatasetFetcher initialization."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)
    assert fetcher.base_dir.exists()
    assert fetcher.config_file == config_file


def test_dataset_fetcher_loads_config(tmp_path: Path) -> None:
    """Test that DatasetFetcher loads configuration correctly."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    available = fetcher.list_available_datasets()
    assert len(available) == 2
    assert available[0].id == "test-dataset"


def test_get_dataset_config(tmp_path: Path) -> None:
    """Test getting a specific dataset configuration."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    config = fetcher.get_dataset_config("test-dataset")
    assert config is not None
    assert config.name == "Test Dataset"

    missing = fetcher.get_dataset_config("nonexistent")
    assert missing is None


def test_list_datasets_empty(tmp_path: Path) -> None:
    """Test listing datasets when none exist."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)
    datasets = fetcher.list_datasets()
    assert datasets == []


def test_list_datasets(tmp_path: Path) -> None:
    """Test listing datasets."""
    config_file = create_test_config(tmp_path)
    base_dir = tmp_path / "datasets"
    fetcher = DatasetFetcher(base_dir, config_file=config_file)

    # Create some test dataset directories
    (base_dir / "dataset1").mkdir()
    (base_dir / "dataset2").mkdir()

    datasets = fetcher.list_datasets()
    assert len(datasets) == 2
    assert "dataset1" in datasets
    assert "dataset2" in datasets


def test_get_dataset_info(tmp_path: Path) -> None:
    """Test getting dataset information."""
    config_file = create_test_config(tmp_path)
    base_dir = tmp_path / "datasets"
    fetcher = DatasetFetcher(base_dir, config_file=config_file)

    # Create a test dataset with some images
    dataset_dir = base_dir / "test_dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "image1.png").touch()
    (dataset_dir / "image2.jpg").touch()
    (dataset_dir / "subdir").mkdir()
    (dataset_dir / "subdir" / "image3.png").touch()

    info = fetcher.get_dataset_info("test_dataset")
    assert info is not None
    assert info["path"] == dataset_dir
    assert info["image_count"] == 3


def test_get_dataset_info_nonexistent(tmp_path: Path) -> None:
    """Test getting info for non-existent dataset."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)
    info = fetcher.get_dataset_info("nonexistent")
    assert info is None


def test_extract_archive_zip(tmp_path: Path) -> None:
    """Test extracting a ZIP archive."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    # Create a test ZIP archive
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("test_file.txt", "test content")
        zf.writestr("subdir/another_file.txt", "more content")

    # Extract it
    extract_dir = tmp_path / "extracted"
    result = fetcher.extract_archive(archive_path, extract_dir)

    assert result is True
    assert (extract_dir / "test_file.txt").exists()
    assert (extract_dir / "subdir" / "another_file.txt").exists()


def test_extract_archive_invalid_format(tmp_path: Path) -> None:
    """Test extracting an unsupported archive format."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    # Create a fake archive with unsupported extension
    archive_path = tmp_path / "test.rar"
    archive_path.touch()

    extract_dir = tmp_path / "extracted"
    result = fetcher.extract_archive(archive_path, extract_dir)

    assert result is False


@patch("src.dataset.requests.get")
def test_download_file_success(mock_get: Mock, tmp_path: Path) -> None:
    """Test successful file download."""
    config_file = create_test_config(tmp_path)

    # Mock the response
    mock_response = Mock()
    mock_response.headers.get.return_value = "1024"
    mock_response.iter_content.return_value = [b"test" * 256]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)
    output_path = tmp_path / "test_file.txt"

    result = fetcher.download_file("http://example.com/file.txt", output_path)

    assert result is True
    assert output_path.exists()


@patch("src.dataset.requests.get")
def test_download_file_failure(mock_get: Mock, tmp_path: Path) -> None:
    """Test failed file download."""
    config_file = create_test_config(tmp_path)

    # Mock the response to raise an exception
    mock_get.side_effect = Exception("Network error")

    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)
    output_path = tmp_path / "test_file.txt"

    result = fetcher.download_file("http://example.com/file.txt", output_path)

    assert result is False
    assert not output_path.exists()


def test_fetch_div2k_valid_mapping(tmp_path: Path) -> None:
    """Test that fetch_div2k correctly maps to fetch_dataset."""
    config_file = create_test_config(tmp_path)
    base_dir = tmp_path / "datasets"
    fetcher = DatasetFetcher(base_dir, config_file=config_file)

    # Create a fake existing DIV2K dataset
    dataset_dir = base_dir / "DIV2K_valid"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "image1.png").touch()
    (dataset_dir / "image2.png").touch()

    result = fetcher.fetch_div2k(split="valid")

    assert result == dataset_dir
    assert result.exists()


def test_fetch_dataset_already_exists(tmp_path: Path) -> None:
    """Test fetching a dataset when it already exists."""
    config_file = create_test_config(tmp_path)
    base_dir = tmp_path / "datasets"
    fetcher = DatasetFetcher(base_dir, config_file=config_file)

    # Create a fake existing dataset
    dataset_dir = base_dir / "test_dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "image1.png").touch()
    (dataset_dir / "image2.png").touch()

    result = fetcher.fetch_dataset("test-dataset")

    assert result == dataset_dir
    assert result.exists()


def test_fetch_dataset_invalid_id(tmp_path: Path) -> None:
    """Test fetching a dataset with invalid ID."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    result = fetcher.fetch_dataset("nonexistent-dataset")

    assert result is None


@patch("src.dataset.gdown")
def test_download_from_google_drive_file(mock_gdown: Mock, tmp_path: Path) -> None:
    """Test downloading a file from Google Drive."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    output_path = tmp_path / "test.zip"
    mock_gdown.download.return_value = str(output_path)

    # Create the file to simulate successful download
    output_path.touch()

    result = fetcher.download_from_google_drive(
        "https://drive.google.com/file/d/test123/view", output_path, is_folder=False
    )

    assert result is True
    mock_gdown.download.assert_called_once()


@patch("src.dataset.gdown")
def test_download_from_google_drive_folder(mock_gdown: Mock, tmp_path: Path) -> None:
    """Test downloading a folder from Google Drive."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    output_path = tmp_path / "test_folder"
    mock_gdown.download_folder.return_value = ["file1.png", "file2.png"]

    # Create the folder to simulate successful download
    output_path.mkdir()

    result = fetcher.download_from_google_drive(
        "https://drive.google.com/drive/folders/test123", output_path, is_folder=True
    )

    assert result is True
    mock_gdown.download_folder.assert_called_once()


@patch("src.dataset.gdown")
def test_download_from_google_drive_failure(mock_gdown: Mock, tmp_path: Path) -> None:
    """Test Google Drive download failure handling."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    output_path = tmp_path / "test.zip"
    mock_gdown.download.side_effect = Exception("Download failed")

    result = fetcher.download_from_google_drive(
        "https://drive.google.com/file/d/test123/view", output_path
    )

    assert result is False


@patch("src.dataset.requests.get")
def test_download_from_dropbox(mock_get: Mock, tmp_path: Path) -> None:
    """Test downloading from Dropbox."""
    config_file = create_test_config(tmp_path)

    # Mock successful download
    mock_response = Mock()
    mock_response.headers.get.return_value = "1024"
    mock_response.iter_content.return_value = [b"test" * 256]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)
    output_path = tmp_path / "test.zip"

    # Test with dl=0 URL
    result = fetcher.download_from_dropbox(
        "https://www.dropbox.com/s/test123/file.zip?dl=0", output_path
    )

    assert result is True
    # Verify URL was converted to dl=1
    called_url = mock_get.call_args[0][0]
    assert "dl=1" in called_url
    assert "dl=0" not in called_url


def test_dropbox_url_conversion_no_params(tmp_path: Path) -> None:
    """Test Dropbox URL conversion when URL has no parameters."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    with patch.object(fetcher, "download_file") as mock_download:
        mock_download.return_value = True
        output_path = tmp_path / "test.zip"

        fetcher.download_from_dropbox("https://www.dropbox.com/s/test123/file.zip", output_path)

        # Verify URL was modified to add dl=1
        called_url = mock_download.call_args[0][0]
        assert "?dl=1" in called_url


def test_dropbox_url_conversion_with_params(tmp_path: Path) -> None:
    """Test Dropbox URL conversion when URL already has parameters."""
    config_file = create_test_config(tmp_path)
    fetcher = DatasetFetcher(tmp_path / "datasets", config_file=config_file)

    with patch.object(fetcher, "download_file") as mock_download:
        mock_download.return_value = True
        output_path = tmp_path / "test.zip"

        fetcher.download_from_dropbox(
            "https://www.dropbox.com/s/test123/file.zip?key=value", output_path
        )

        # Verify URL was modified to add &dl=1
        called_url = mock_download.call_args[0][0]
        assert "&dl=1" in called_url

