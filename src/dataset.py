"""Dataset fetching and management module.

This module handles downloading and organizing test image datasets
for the research project. It provides an extensible architecture for
supporting multiple dataset sources through a JSON configuration file.
"""

import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gdown
import requests
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    id: str
    name: str
    description: str
    type: str
    url: str
    size_mb: float
    image_count: int
    resolution: str
    format: str
    extracted_folder: str | None = None
    rename_to: str | None = None
    license: str | None = None
    source: str | None = None
    storage_type: str = "direct"  # direct, google_drive, dropbox, dropbox_folder
    folder_id: str | None = None  # For Google Drive folder downloads
    post_process: str | None = None  # Post-processing action: extract_multipart_zips

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        """Create DatasetConfig from dictionary.

        Args:
            data: Dictionary with dataset configuration

        Returns:
            DatasetConfig instance
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            type=data["type"],
            url=data["url"],
            size_mb=data["size_mb"],
            image_count=data["image_count"],
            resolution=data["resolution"],
            format=data["format"],
            extracted_folder=data.get("extracted_folder"),
            rename_to=data.get("rename_to"),
            license=data.get("license"),
            source=data.get("source"),
            storage_type=data.get("storage_type", "direct"),
            folder_id=data.get("folder_id"),
            post_process=data.get("post_process"),
        )


class DatasetFetcher:
    """Handles fetching and managing image datasets.

    This class provides methods for downloading datasets from various sources,
    extracting archives, and managing the local dataset storage. It uses a
    JSON configuration file to define available datasets.
    """

    def __init__(self, base_dir: Path, config_file: Path | None = None) -> None:
        """Initialize the dataset fetcher.

        Args:
            base_dir: Base directory where datasets will be stored
            config_file: Path to datasets.json configuration file.
                        If None, looks for config/datasets.json.
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_file is None:
            # Default to config/datasets.json
            project_root = Path(__file__).parent.parent
            config_file = project_root / "config" / "datasets.json"

        self.config_file = config_file
        self._datasets: dict[str, DatasetConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load dataset configuration from JSON file."""
        if not self.config_file.exists():
            print(f"Warning: Config file not found: {self.config_file}")
            return

        try:
            with open(self.config_file) as f:
                config_data = json.load(f)

            for dataset_dict in config_data.get("datasets", []):
                dataset = DatasetConfig.from_dict(dataset_dict)
                self._datasets[dataset.id] = dataset

        except Exception as e:
            print(f"Error loading config file {self.config_file}: {e}")

    def list_available_datasets(self) -> list[DatasetConfig]:
        """List all datasets available in configuration.

        Returns:
            List of DatasetConfig objects for available datasets
        """
        return list(self._datasets.values())

    def get_dataset_config(self, dataset_id: str) -> DatasetConfig | None:
        """Get configuration for a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetConfig if found, None otherwise
        """
        return self._datasets.get(dataset_id)

    def download_file(self, url: str, output_path: Path, description: str | None = None) -> bool:
        """Download a file from URL with progress bar.

        Args:
            url: URL of the file to download
            output_path: Path where the file will be saved
            description: Optional description for the progress bar

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            output_path.parent.mkdir(parents=True, exist_ok=True)

            desc = description or output_path.name
            with (
                open(output_path, "wb") as f,
                tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False

    def download_from_google_drive(
        self, url: str, output_path: Path, description: str | None = None, is_folder: bool = False
    ) -> bool:
        """Download a file or folder from Google Drive.

        Args:
            url: Google Drive URL or file ID
            output_path: Path where the file/folder will be saved
            description: Optional description for download
            is_folder: Whether this is a folder download

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            desc = description or "Downloading from Google Drive"
            print(f"{desc}...")

            if is_folder:
                # Download entire folder
                gdown.download_folder(url, output=str(output_path), quiet=False, use_cookies=False)
            else:
                # Download single file
                gdown.download(url, str(output_path), quiet=False, fuzzy=True)

            return output_path.exists()
        except Exception as e:
            print(f"Failed to download from Google Drive: {e}")
            print("Note: Google Drive downloads may fail due to access restrictions.")
            print("Consider downloading manually if automatic download fails.")
            return False

    def download_from_dropbox(
        self, url: str, output_path: Path, description: str | None = None
    ) -> bool:
        """Download a file from Dropbox.

        Args:
            url: Dropbox sharing URL
            output_path: Path where the file will be saved
            description: Optional description for the progress bar

        Returns:
            True if download succeeded, False otherwise
        """
        # Convert Dropbox sharing link to direct download link
        # Change dl=0 to dl=1 or add ?dl=1
        if "dl=0" in url:
            direct_url = url.replace("dl=0", "dl=1")
        elif "?" in url:
            direct_url = url + "&dl=1"
        else:
            direct_url = url + "?dl=1"

        return self.download_file(direct_url, output_path, description)

    def download_image(self, url: str, output_path: Path) -> bool:
        """Download a single image from URL.

        Args:
            url: URL of the image to download
            output_path: Path where the image will be saved

        Returns:
            True if download succeeded, False otherwise
        """
        return self.download_file(url, output_path)

    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """Extract a ZIP or TAR archive.

        Args:
            archive_path: Path to the archive file
            extract_dir: Directory where contents will be extracted

        Returns:
            True if extraction succeeded, False otherwise
        """
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)

            if archive_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    # Get list of files for progress bar
                    members = zip_ref.namelist()
                    with tqdm(total=len(members), desc=f"Extracting {archive_path.name}") as pbar:
                        for member in members:
                            zip_ref.extract(member, extract_dir)
                            pbar.update(1)

            elif archive_path.suffix.lower() in [".tar", ".gz", ".tgz"]:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_members = tar_ref.getmembers()
                    with tqdm(
                        total=len(tar_members), desc=f"Extracting {archive_path.name}"
                    ) as pbar:
                        for tar_member in tar_members:
                            tar_ref.extract(tar_member, extract_dir)
                            pbar.update(1)
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False

            return True
        except Exception as e:
            print(f"Failed to extract {archive_path}: {e}")
            return False

    def extract_multipart_zips(self, dataset_dir: Path) -> bool:
        """Extract multi-part zip archives in a directory.

        LIU4K v2 datasets use multi-part zips (.zip, .z01, .z02, etc.) organized by category.
        This method finds and extracts all such archives.

        Args:
            dataset_dir: Directory containing multi-part zip files

        Returns:
            True if all extractions succeeded, False otherwise
        """
        try:
            import subprocess

            # Find all .zip files (the last part of multi-part archives)
            zip_files = sorted(dataset_dir.glob("*.zip"))

            if not zip_files:
                print(f"No .zip files found in {dataset_dir}")
                return True  # Not an error if no zips to extract

            print(f"Found {len(zip_files)} multi-part archives to extract")

            for zip_file in zip_files:
                category_name = zip_file.stem  # e.g., "Animal" from "Animal.zip"
                print(f"Extracting {category_name}...")

                # Extract to a subdirectory named after the category
                extract_to = dataset_dir / category_name
                extract_to.mkdir(exist_ok=True)

                # Use 7z to extract multi-part archives
                # 7z automatically handles .z01, .z02, etc. when you point it to .zip
                cmd = ["7z", "x", "-y", f"-o{extract_to}", str(zip_file)]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    print(f"✗ Failed to extract {category_name}")
                    print(f"Error: {result.stderr}")
                    return False

                # Count extracted files
                extracted_files = list(extract_to.rglob("*"))
                extracted_count = len([f for f in extracted_files if f.is_file()])
                print(f"✓ Extracted {extracted_count} files to {category_name}/")

            # Clean up archive files after successful extraction
            print("\nCleaning up archive files...")
            archive_files = list(dataset_dir.glob("*.zip")) + list(dataset_dir.glob("*.z*"))
            for archive_file in archive_files:
                archive_file.unlink()
                print(f"  Removed {archive_file.name}")
            print(f"✓ Removed {len(archive_files)} archive file(s)")

            return True
        except Exception as e:
            print(f"Failed to extract multi-part zips: {e}")
            return False

    def extract_zips(self, dataset_dir: Path) -> bool:
        """Extract single-file zip archives in a directory.

        LIU4K v1 datasets use single zip files containing all images.
        This method finds and extracts all such archives.

        Args:
            dataset_dir: Directory containing zip files

        Returns:
            True if all extractions succeeded, False otherwise
        """
        try:
            import zipfile

            # Find all .zip files
            zip_files = sorted(dataset_dir.glob("*.zip"))

            if not zip_files:
                print(f"No .zip files found in {dataset_dir}")
                return True  # Not an error if no zips to extract

            print(f"Found {len(zip_files)} archive(s) to extract")

            for zip_file in zip_files:
                archive_name = zip_file.stem
                print(f"Extracting {archive_name}...")

                # Extract directly to the dataset directory
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    members = zip_ref.namelist()
                    with tqdm(total=len(members), desc=f"Extracting {archive_name}") as pbar:
                        for member in members:
                            zip_ref.extract(member, dataset_dir)
                            pbar.update(1)

                # Count extracted files
                image_extensions = [".png", ".jpg", ".jpeg"]
                image_files = []
                for ext in image_extensions:
                    image_files.extend(list(dataset_dir.rglob(f"*{ext}")))

                print(f"✓ Extracted {len(members)} items, found {len(image_files)} images")

            # Clean up archive files after successful extraction
            print("\nCleaning up archive files...")
            for zip_file in zip_files:
                zip_file.unlink()
                print(f"  Removed {zip_file.name}")
            print(f"✓ Removed {len(zip_files)} archive file(s)")

            return True
        except Exception as e:
            print(f"Failed to extract zip archives: {e}")
            return False

    def fetch_dataset(
        self,
        dataset_id: str,
        cleanup_archive: bool = True,
    ) -> Path | None:
        """Fetch a dataset using its configuration.

        Args:
            dataset_id: Dataset identifier from datasets.json
            cleanup_archive: Whether to delete the downloaded archive after extraction

        Returns:
            Path to the extracted dataset directory, or None if fetch failed
        """
        # Get dataset configuration
        config = self.get_dataset_config(dataset_id)
        if config is None:
            print(f"Error: Dataset '{dataset_id}' not found in configuration")
            available = [d.id for d in self.list_available_datasets()]
            print(f"Available datasets: {', '.join(available)}")
            return None

        # Determine final dataset directory name
        dataset_name = config.rename_to or dataset_id
        dataset_dir = self.base_dir / dataset_name

        # Check if dataset already exists
        if dataset_dir.exists():
            image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(dataset_dir.rglob(f"*{ext}")))

            if image_files:
                print(f"Dataset '{config.name}' already exists at {dataset_dir}")
                print(f"Found {len(image_files)} images")
                return dataset_dir

        # Determine archive filename from URL
        url_path = config.url.split("/")[-1]
        if not url_path or config.storage_type == "google_drive":
            # For Google Drive or unclear URLs, use dataset ID as filename
            if config.type == "zip":
                url_path = f"{dataset_id}.zip"
            else:
                url_path = f"{dataset_id}.{config.type}"
        archive_path = self.base_dir / url_path

        # Download archive
        print(f"Downloading {config.name}...")
        print(f"  Size: ~{config.size_mb} MB")
        print(f"  Images: {config.image_count}")
        print(f"  Format: {config.format}")
        if config.storage_type != "direct":
            print(f"  Storage: {config.storage_type}")
        print()

        # Use appropriate download method based on storage type
        download_success = False
        if config.storage_type == "google_drive":
            if config.folder_id:
                # Download entire folder
                download_success = self.download_from_google_drive(
                    config.url, dataset_dir, description=config.name, is_folder=True
                )
                # For folder downloads, handle post-processing and verification
                if download_success and dataset_dir.exists():
                    # Apply post-processing if specified
                    if config.post_process == "extract_multipart_zips":
                        print("\nPost-processing: Extracting multi-part zip archives...")
                        if not self.extract_multipart_zips(dataset_dir):
                            print(
                                "Warning: Post-processing failed, but dataset may still be usable"
                            )
                    elif config.post_process == "extract_zips":
                        print("\nPost-processing: Extracting zip archives...")
                        if not self.extract_zips(dataset_dir):
                            print(
                                "Warning: Post-processing failed, but dataset may still be usable"
                            )

                    # Verify images
                    image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl"]
                    image_files = []
                    for ext in image_extensions:
                        image_files.extend(list(dataset_dir.rglob(f"*{ext}")))

                    print(f"\nSuccessfully fetched {config.name}")
                    print(f"Location: {dataset_dir}")
                    print(f"Images: {len(image_files)}")
                    return dataset_dir
            else:
                # Download single archive file
                download_success = self.download_from_google_drive(
                    config.url, archive_path, description=config.name
                )
        elif config.storage_type == "dropbox":
            download_success = self.download_from_dropbox(
                config.url, archive_path, description=config.name
            )
        else:  # direct URL
            download_success = self.download_file(config.url, archive_path, description=config.name)

        if not download_success:
            return None

        # Extract archive
        print(f"Extracting {url_path}...")
        if not self.extract_archive(archive_path, self.base_dir):
            return None

        # Handle folder renaming if needed
        if config.extracted_folder:
            extracted_path = self.base_dir / config.extracted_folder
            if extracted_path.exists() and extracted_path != dataset_dir:
                extracted_path.rename(dataset_dir)
        else:
            # Try to find the extracted folder
            # This handles cases where the archive creates a top-level folder
            potential_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and d != dataset_dir]
            if len(potential_dirs) == 1:
                potential_dirs[0].rename(dataset_dir)

        # Verify dataset exists
        if not dataset_dir.exists():
            print(f"Warning: Expected folder {dataset_dir} not found after extraction")
            return None

        # Cleanup archive if requested
        if cleanup_archive and archive_path.exists():
            archive_path.unlink()
            print(f"Removed archive: {url_path}")

        # Verify images
        image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(dataset_dir.rglob(f"*{ext}")))

        print(f"Successfully fetched {config.name}")
        print(f"Location: {dataset_dir}")
        print(f"Images: {len(image_files)}")

        return dataset_dir

    def fetch_div2k(
        self,
        split: Literal["train", "valid"] = "valid",
        cleanup_archive: bool = True,
    ) -> Path | None:
        """Fetch the DIV2K dataset.

        This is a convenience method that maps to the configuration-based fetch.

        Args:
            split: Dataset split to download - "train" or "valid"
            cleanup_archive: Whether to delete the downloaded archive after extraction

        Returns:
            Path to the extracted dataset directory, or None if fetch failed
        """
        dataset_id = f"div2k-{split}"
        return self.fetch_dataset(dataset_id, cleanup_archive)

    def list_datasets(self) -> list[str]:
        """List all available datasets.

        Returns:
            List of dataset names (directory names in the base directory)
        """
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def get_dataset_info(self, dataset_name: str) -> dict[str, int | Path] | None:
        """Get information about a downloaded dataset.

        Args:
            dataset_name: Name of the dataset directory

        Returns:
            Dictionary with dataset information (path, image count), or None if not found
        """
        dataset_dir = self.base_dir / dataset_name
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            return None

        # Count images (common formats)
        image_extensions = [".png", ".jpg", ".jpeg", ".webp", ".avif", ".jxl"]
        image_count = sum(len(list(dataset_dir.rglob(f"*{ext}"))) for ext in image_extensions)

        return {
            "path": dataset_dir,
            "image_count": image_count,
        }
