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
- **type** (required): Archive type (`zip`, `tar`, `tar.gz`, `tgz`)
- **url** (required): Download URL
- **size_mb**: Approximate size in megabytes
- **image_count**: Number of images
- **resolution**: Resolution description (e.g., "2K", "4K")
- **format**: Image format (e.g., "PNG", "JPEG")
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

**Validation Set:**
- Images: 100
- Size: ~449MB
- URL: `http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip`

**Training Set:**
- Images: 800
- Size: ~3.53GB
- URL: `http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip`

### Storage Location

Datasets are stored in the `data/datasets/` directory:

```text
data/datasets/
├── DIV2K_valid/         # Validation set (100 images)
│   ├── 0801.png
│   ├── 0802.png
│   └── ...
└── DIV2K_train/         # Training set (800 images)
    ├── 0001.png
    ├── 0002.png
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

### Download Fails

If the download fails, check:
1. Your internet connection
2. The dataset URL is still accessible
3. You have sufficient disk space

You can retry the download - the fetcher will resume if possible.

### Insufficient Disk Space

**Validation set**: Requires ~1GB free space (500MB download + extraction)

**Training set**: Requires ~8GB free space (4GB download + extraction)

To save space, enable archive cleanup (default behavior).

### Slow Downloads

The download uses a progress bar to show speed and estimated time. If downloads are slow:
1. Check your network connection
2. Try downloading during off-peak hours
3. Consider using a different network

## Future Dataset Support

The configuration-based architecture makes it easy to add new datasets. Simply add entries to `config/datasets.json`:

- **Flickr2K**: High-quality 2K images from Flickr
- **HuggingFace Datasets**: Integration with HF's dataset hub  
- **Custom URLs**: Download from any HTTP/HTTPS source
- **High-resolution datasets**: 4K or higher resolution images

See [Dataset Support and Roadmap](../reference/datasets.md) for more information about planned datasets and evaluation criteria.

