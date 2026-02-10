#!/usr/bin/env python3
"""Script to fetch datasets for the research project.

This script provides a command-line interface for downloading
and managing image datasets using configuration from datasets.json.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import DatasetFetcher


def list_datasets(fetcher: DatasetFetcher) -> None:
    """List all available datasets from configuration.

    Args:
        fetcher: DatasetFetcher instance
    """
    print("=" * 70)
    print("Available Datasets")
    print("=" * 70)
    print()

    available = fetcher.list_available_datasets()
    if not available:
        print("No datasets configured.")
        return

    for dataset in available:
        print(f"ID: {dataset.id}")
        print(f"  Name: {dataset.name}")
        print(f"  Description: {dataset.description}")
        print(f"  Size: ~{dataset.size_mb} MB")
        print(f"  Images: {dataset.image_count}")
        print(f"  Resolution: {dataset.resolution}")
        print(f"  Format: {dataset.format}")
        if dataset.source:
            print(f"  Source: {dataset.source}")
        print()


def list_downloaded(fetcher: DatasetFetcher) -> None:
    """List all downloaded datasets.

    Args:
        fetcher: DatasetFetcher instance
    """
    existing = fetcher.list_datasets()
    if existing:
        print("Downloaded datasets:")
        for dataset_name in existing:
            info = fetcher.get_dataset_info(dataset_name)
            if info:
                print(f"  - {dataset_name}: {info['image_count']} images")
    else:
        print("No datasets downloaded yet.")


def fetch_dataset(fetcher: DatasetFetcher, dataset_id: str, keep_archive: bool) -> int:
    """Fetch a specific dataset.

    Args:
        fetcher: DatasetFetcher instance
        dataset_id: Dataset identifier to fetch
        keep_archive: Whether to keep the archive after extraction

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 70)
    print("Dataset Fetcher")
    print("=" * 70)
    print()

    dataset_path = fetcher.fetch_dataset(dataset_id, cleanup_archive=not keep_archive)

    if dataset_path:
        print()
        print("=" * 70)
        print("✓ Dataset ready!")
        print(f"  Location: {dataset_path}")
        print("=" * 70)
        return 0
    else:
        print()
        print("=" * 70)
        print("✗ Failed to fetch dataset")
        print("=" * 70)
        return 1


def main() -> int:
    """Main entry point for the dataset fetching script.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Fetch and manage image datasets for web image formats research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  %(prog)s --list

  # Fetch DIV2K validation dataset
  %(prog)s div2k-valid

  # Fetch and keep the archive
  %(prog)s div2k-train --keep-archive

  # Show downloaded datasets
  %(prog)s --show-downloaded
        """,
    )

    parser.add_argument(
        "dataset_id",
        nargs="?",
        help="Dataset ID to fetch (from datasets.json)",
    )

    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all available datasets from configuration",
    )

    parser.add_argument(
        "-s",
        "--show-downloaded",
        action="store_true",
        help="Show already downloaded datasets",
    )

    parser.add_argument(
        "-k",
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive after extraction (default: delete)",
    )

    parser.add_argument(
        "-d",
        "--datasets-dir",
        type=Path,
        help="Directory for storing datasets (default: ./data/datasets)",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to datasets.json configuration file (default: ./config/datasets.json)",
    )

    args = parser.parse_args()

    # Determine base directory
    base_dir = args.datasets_dir or Path(__file__).parent.parent / "data" / "datasets"

    # Determine config file
    config_file = args.config if args.config else None

    # Initialize fetcher
    fetcher = DatasetFetcher(base_dir, config_file=config_file)

    # Handle different commands
    if args.list:
        list_datasets(fetcher)
        return 0

    if args.show_downloaded:
        list_downloaded(fetcher)
        return 0

    if not args.dataset_id:
        parser.print_help()
        print()
        print("Tip: Use --list to see available datasets")
        return 1

    return fetch_dataset(fetcher, args.dataset_id, args.keep_archive)


if __name__ == "__main__":
    sys.exit(main())
