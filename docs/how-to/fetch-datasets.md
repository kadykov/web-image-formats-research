# How to Fetch Datasets

This guide explains how to download and manage image datasets for the research project using the configuration-based fetching system.

## Quick Start

### List Available Datasets

```bash
just list-available-datasets
```

This shows all datasets defined in `config/datasets.json`.

### Fetch DIV2K Validation Dataset (Recommended)

For development and testing, start with the validation dataset:

```bash
just fetch div2k-valid
```

This downloads 100 high-quality 2K resolution images (~449MB).

### Fetch DIV2K Training Dataset

For comprehensive research, use the full training dataset:

```bash
just fetch div2k-train
```

This downloads 800 high-quality 2K resolution images (~3.53GB).

### Fetch High-Resolution Datasets (4K-6K)

For higher resolution research, we support the LIU4K benchmark datasets:

#### LIU4K v1 Validation ⭐ **Recommended for 4K research**

```bash
just fetch liu4k-v1-valid
```

This downloads and extracts 80 high-quality 4K lossless PNG images (~1.26GB zip) from Google Drive. The system automatically extracts the single zip archive.

**Why v1?** More manageable size, single-file zip (simpler than v2's multi-part archives).

#### LIU4K v1 Training

```bash
just fetch liu4k-v1-train
```

This downloads and extracts 800 high-quality 4K lossless PNG images (~10GB zip) from Google Drive. The system automatically extracts the single zip archive.

#### LIU4K v2 Validation (Advanced)

```bash
just fetch liu4k-v2-valid
```

This downloads 400 high-quality 4K-6K lossless PNG images (~15GB) from Google Drive, stored as multi-part zip archives by category (Animal, Building, Capture, Mountain, Street). The system automatically extracts the archives.

#### LIU4K v2 Training (Advanced)

```bash
just fetch liu4k-v2-train
```

This downloads 1600 high-quality 4K-6K lossless PNG images (~60GB) from Google Drive, stored as multi-part zip archives.

**Note:**

- All LIU4K datasets use **CC BY-NC-ND 4.0** license (attribution required, non-commercial, no derivatives)
- Google Drive downloads may occasionally fail due to access restrictions or quota limits
- **LIU4K v1** downloads as a single zip archive (automatically extracted with Python's zipfile)
- **LIU4K v2** downloads as multi-part zip archives (automatically extracted with 7z)
- If automatic download fails, manually download from Google Drive and place in `data/datasets/`

**System Requirement for v2:** LIU4K v2 datasets use multi-part zip archives (.zip, .z01, .z02, etc.) and require **7z** to be installed. The dev container includes 7z by default. If running outside the container, install with:

```bash
# Debian/Ubuntu
sudo apt-get install p7zip-full

# macOS
brew install p7zip
```

**Note:** LIU4K v1 uses standard single-file zips and doesn't require 7z.

#### UHD-IQA Full Dataset

```bash
just fetch uhd-iqa-full
```

This downloads 6073 4K JPEG images with quality ratings (~10.7GB). Note: JPEG format (lossy).

**Note:** Google Drive downloads may occasionally fail due to access restrictions or quota limits. If automatic download fails, you can manually download from the source and place in `data/datasets/`.

**LIU4K License:** CC BY-NC-ND 4.0 (requires attribution, non-commercial use only, no derivatives)

### Check Downloaded Datasets

```bash
just list-datasets
```

## Using the Command Line Script

The `fetch_dataset.py` script provides a flexible CLI for dataset management.

### List Available Datasets

```bash
python scripts/fetch_dataset.py --list
```

### Fetch a Dataset

```bash
python scripts/fetch_dataset.py div2k-valid
```

### Keep Archive After Download

By default, archives are deleted after extraction to save space:

```bash
python scripts/fetch_dataset.py div2k-train --keep-archive
```

### Show Downloaded Datasets

```bash
python scripts/fetch_dataset.py --show-downloaded
```

### Custom Datasets Directory

```bash
python scripts/fetch_dataset.py div2k-valid --datasets-dir /path/to/datasets
```

### Custom Configuration File

```bash
python scripts/fetch_dataset.py div2k-valid --config /path/to/datasets.json
```

## Using the Python API

### Basic Usage

```python
from pathlib import Path
from src.dataset import DatasetFetcher

# Initialize fetcher (uses config/datasets.json by default)
fetcher = DatasetFetcher(Path("data/datasets"))

# Fetch validation dataset by ID
dataset_path = fetcher.fetch_dataset("div2k-valid")
print(f"Dataset ready at: {dataset_path}")

# Fetch training dataset
dataset_path = fetcher.fetch_dataset("div2k-train")
```

### List Available Datasets from Configuration

```python
# List all datasets defined in config/datasets.json
available = fetcher.list_available_datasets()
for dataset in available:
    print(f"{dataset.id}: {dataset.name}")
    print(f"  Size: {dataset.size_mb} MB")
    print(f"  Images: {dataset.image_count}")
```

### Check Existing Datasets

```python
# List all downloaded datasets
datasets = fetcher.list_datasets()
for dataset_name in datasets:
    info = fetcher.get_dataset_info(dataset_name)
    print(f"{dataset_name}: {info['image_count']} images")
```

### Using the Old API (Still Supported)

The legacy `fetch_div2k()` method still works for backwards compatibility:

```python
# This maps to fetch_dataset("div2k-valid")
dataset_path = fetcher.fetch_div2k(split="valid")
```

## Configuration File

Datasets are configured in `config/datasets.json`. Each dataset entry includes:

```json
{
  "datasets": [
    {
      "id": "div2k-valid",
      "name": "DIV2K Validation",
      "description": "DIV2K validation set with 100 high-quality 2K images",
      "type": "zip",
      "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
      "size_mb": 449,
      "image_count": 100,
      "resolution": "2K",
      "format": "PNG",
      "extracted_folder": "DIV2K_valid_HR",
      "rename_to": "DIV2K_valid",
      "license": "Unknown - check DIV2K website",
      "source": "ETH Zurich Computer Vision Lab"
    }
  ]
}
```

### Configuration Schema

The configuration follows the JSON schema defined in `config/datasets.schema.json`:

- **id** (required): Unique identifier (lowercase, hyphens allowed)
- **name** (required): Human-readable name
- **description**: Brief description
- **type** (required): Archive type (`zip`, `tar`, `tar.gz`, `tgz`) or `folder` for Google Drive folders
- **url** (required): Download URL (supports direct URLs, Google Drive, Dropbox)
- **size_mb**: Approximate size in megabytes
- **image_count**: Number of images
- **resolution**: Resolution description (e.g., "2K", "4K")
- **format**: Image format (e.g., "PNG", "JPEG")
- **extracted_folder**: Folder name after extraction
- **rename_to**: Rename extracted folder to this name
- **license**: License information
- **source**: Organization or project providing the dataset
- **storage_type** (optional): Storage provider - `direct` (default), `google_drive`, or `dropbox`
- **folder_id** (optional): Google Drive folder ID for folder downloads
- **post_process** (optional): Post-processing action - `extract_multipart_zips` for LIU4K v2
- **extracted_folder**: Folder name after extraction
- **rename_to**: Rename extracted folder to this name
- **license**: License information
- **source**: Organization or project providing the dataset

### Adding a New Dataset

To add a new dataset:

1. Edit `config/datasets.json`
2. Add a new entry with all required fields
3. Run `python scripts/fetch_dataset.py --list` to verify
4. Fetch with `just fetch <dataset-id>`

Example:

```json
{
  "id": "my-dataset",
  "name": "My Custom Dataset",
  "description": "A custom dataset for testing",
  "type": "zip",
  "url": "https://example.com/dataset.zip",
  "size_mb": 1000,
  "image_count": 200,
  "resolution": "4K",
  "format": "PNG"
}
```

## Dataset Details

### DIV2K

- **Source**: ETH Zurich Computer Vision Lab
- **Resolution**: 2K (variable aspect ratios)
- **Format**: PNG (uncompressed)
- **Quality**: High-quality photographs
- **License**: Unknown - check DIV2K website

**Validation Set:**

- Images: 100
- Size: ~449MB
- URL: `http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip`

**Training Set:**

- Images: 800
- Size: ~3.53GB
- URL: `http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip`

### LIU4K-v2 (Large-scale Ideal UHD 4K)

- **Source**: Peking University
- **Resolution**: 4K-6K (most images between 4K-6K, minimum 3K)
- **Format**: PNG (lossless)
- **Quality**: Very high-quality diverse photographs
- **License**: CC0 (Public Domain)
- **Storage**: Google Drive (requires gdown)

**Validation Set (Recommended for 4K research):**

- Images: 400
- Size: ~1.26GB
- URL: Google Drive folder
- Best for testing and validation with manageable size

**Training Set:**

- Images: 1600
- Size: ~5GB
- URL: Google Drive folder
- Comprehensive training set for 4K research

**Note:** This dataset is ideal for high-resolution format research as it provides lossless PNG images at resolutions between 4K-6K, significantly higher than DIV2K (2K).

### UHD-IQA Benchmark

- **Source**: University of Konstanz
- **Resolution**: UHD-1 (3840x2160 = 4K)
- **Format**: JPEG (lossy, from Pixabay)
- **Quality**: High-quality photos with quality ratings
- **License**: CC0 (Public Domain)
- **Storage**: Direct download

**Full Dataset:**

- Images: 6,073
- Size: ~10.7GB
- URL: `https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/UHD-IQA-database.zip`
- Includes quality ratings (MOS) and metadata
- Split into training, validation, and test sets

**Note:** While this dataset has JPEG images (lossy), it provides a large number of 4K images with quality ratings, useful for benchmarking and quality analysis.

### Storage Location

Datasets are stored in the `data/datasets/` directory:

```text
data/datasets/
├── DIV2K_valid/         # Validation set (100 2K images)
│   ├── 0801.png
│   ├── 0802.png
│   └── ...
├── DIV2K_train/         # Training set (800 2K images)
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── LIU4K_valid/         # Validation set (400 4K-6K images)
│   └── ...
├── LIU4K_train/         # Training set (1600 4K-6K images)
│   └── ...
└── UHD_IQA/             # Full dataset (6073 4K JPEG images)
    └── ...
```

## Advanced Usage

### Keep Downloaded Archives

By default, archives are deleted after extraction to save space. To keep them:

```python
dataset_path = fetcher.fetch_div2k(split="valid", cleanup_archive=False)
```

### Check Before Downloading

The fetcher automatically checks if a dataset already exists:

```python
# This will use the existing dataset if available
dataset_path = fetcher.fetch_div2k(split="valid")
```

## Troubleshooting

### Dataset Not Found

If you get "Dataset 'xxx' not found in configuration":

```bash
# Check what datasets are available
python scripts/fetch_dataset.py --list
```

### Configuration File Not Found

By default, the script looks for `config/datasets.json`. If you get a warning about a missing config file:

1. Verify `config/datasets.json` exists
2. Or specify a custom path: `--config /path/to/datasets.json`

### Google Drive Download Fails

Google Drive downloads (LIU4K datasets) may fail due to:

1. **Access restrictions**: Google Drive may require authentication for large files
2. **Quota limits**: Google Drive has download quotas that may prevent automated downloads
3. **Network issues**: Large files may timeout

**Solutions:**

- The `gdown` library is automatically installed with dependencies
- Retry the download - sometimes it works on subsequent attempts
- Download manually from the source and place in `data/datasets/`
- Try using a VPN if your region has restrictions

### Download Fails

If the download fails, check:

1. Your internet connection
2. The dataset URL is still accessible
3. You have sufficient disk space
4. For cloud storage, ensure you have the required library (`gdown` for Google Drive)

You can retry the download - the fetcher will resume if possible.

### Insufficient Disk Space

**DIV2K Validation**: Requires ~1GB free space (500MB download + extraction)
**DIV2K Training**: Requires ~8GB free space (4GB download + extraction)
**LIU4K Validation**: Requires ~2.5GB free space (1.3GB download)
**LIU4K Training**: Requires ~10GB free space (5GB download)
**UHD-IQA Full**: Requires ~22GB free space (11GB download + extraction)

To save space, enable archive cleanup (default behavior).

### Slow Downloads

The download uses a progress bar to show speed and estimated time. If downloads are slow:

1. Check your network connection
2. Try downloading during off-peak hours
3. Consider using a different network
4. For Google Drive, downloads may be slower due to API limitations

## Cloud Storage Support

The dataset fetcher now supports multiple storage providers:

### Direct URLs (Default)

Standard HTTP/HTTPS downloads with progress tracking:

```json
{
  "storage_type": "direct",
  "url": "https://example.com/dataset.zip"
}
```

### Google Drive

Supports both single files and folders using the `gdown` library:

```json
{
  "storage_type": "google_drive",
  "type": "folder",
  "url": "https://drive.google.com/drive/folders/FOLDER_ID",
  "folder_id": "FOLDER_ID"
}
```

**Requirements:** `gdown>=5.0.0` (automatically installed)

### Dropbox

Converts sharing links to direct download links:

```json
{
  "storage_type": "dropbox",
  "url": "https://www.dropbox.com/s/xxxx/file.zip?dl=0"
}
```

## Future Dataset Support

The configuration-based architecture makes it easy to add new datasets. We now support:

- ✅ **Direct URLs**: Standard HTTP/HTTPS downloads
- ✅ **Google Drive**: Single files and entire folders
- ✅ **Dropbox**: Sharing links with automatic conversion
- ✅ **High-resolution datasets**: 4K-6K images (LIU4K-v2)
- ✅ **Multiple formats**: ZIP, TAR, TAR.GZ archives

Potential future additions:

- **HuggingFace Datasets**: Integration with HF's dataset hub  
- **AWS S3/Azure**: Cloud storage providers
- **Flickr API**: Direct image downloads
- **8K+ datasets**: Ultra-high resolution for future display technologies
