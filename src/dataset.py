"""Dataset fetching and management module.

This module handles downloading and organizing test image datasets
for the research project.
"""

from pathlib import Path

import requests
from tqdm import tqdm


class DatasetFetcher:
    """Handles fetching and managing image datasets."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize the dataset fetcher.

        Args:
            base_dir: Base directory where datasets will be stored
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_image(self, url: str, output_path: Path) -> bool:
        """Download a single image from URL.

        Args:
            url: URL of the image to download
            output_path: Path where the image will be saved

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with (
                open(output_path, "wb") as f,
                tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False

    def list_datasets(self) -> list[str]:
        """List all available datasets.

        Returns:
            List of dataset names
        """
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
