"""Tests for dataset fetching module."""

from pathlib import Path

from src.dataset import DatasetFetcher


def test_dataset_fetcher_initialization(tmp_path: Path) -> None:
    """Test DatasetFetcher initialization."""
    fetcher = DatasetFetcher(tmp_path / "datasets")
    assert fetcher.base_dir.exists()


def test_list_datasets_empty(tmp_path: Path) -> None:
    """Test listing datasets when none exist."""
    fetcher = DatasetFetcher(tmp_path / "datasets")
    datasets = fetcher.list_datasets()
    assert datasets == []


def test_list_datasets(tmp_path: Path) -> None:
    """Test listing datasets."""
    base_dir = tmp_path / "datasets"
    fetcher = DatasetFetcher(base_dir)

    # Create some test dataset directories
    (base_dir / "dataset1").mkdir()
    (base_dir / "dataset2").mkdir()

    datasets = fetcher.list_datasets()
    assert len(datasets) == 2
    assert "dataset1" in datasets
    assert "dataset2" in datasets
