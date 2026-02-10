# Reference: Configuration Files

This reference describes all configuration files used in the project.

## Location

All configuration files are stored in the `config/` directory.

## datasets.json

Defines available datasets for fetching.

### Schema

Validated by `config/datasets.schema.json`.

### Structure

```json
{
  "datasets": [
    {
      "id": "string",
      "name": "string",
      "description": "string",
      "type": "zip|tar|tar.gz|tgz",
      "url": "string",
      "size_mb": number,
      "image_count": number,
      "resolution": "string",
      "format": "string",
      "extracted_folder": "string (optional)",
      "rename_to": "string (optional)",
      "license": "string (optional)",
      "source": "string (optional)"
    }
  ]
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier. Lowercase with hyphens only. Used in CLI commands. |
| `name` | string | Yes | Human-readable name displayed in listings. |
| `description` | string | Yes | Brief description of the dataset. |
| `type` | string | Yes | Archive format: `zip`, `tar`, `tar.gz`, or `tgz`. |
| `url` | string | Yes | Download URL (HTTP/HTTPS). |
| `size_mb` | number | Yes | Approximate download size in megabytes. |
| `image_count` | number | Yes | Number of images in the dataset. |
| `resolution` | string | Yes | Resolution description (e.g., "2K", "4K", "1080p"). |
| `format` | string | Yes | Image format (e.g., "PNG", "JPEG"). |
| `extracted_folder` | string | No | Name of folder created after extraction. |
| `rename_to` | string | No | Rename extracted folder to this name. |
| `license` | string | No | License information or URL. |
| `source` | string | No | Organization or project providing the dataset. |

### Example

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

### Adding a New Dataset

1. Edit `config/datasets.json`
2. Add a new entry following the schema
3. Verify with `python scripts/fetch_dataset.py --list`
4. Fetch with `just fetch <dataset-id>`

### Validation

The schema `config/datasets.schema.json` enforces:
- Required fields are present
- Types are correct (string, number)
- `id` matches pattern `^[a-z0-9-]+$`
- `type` is one of the allowed values
- `url` is a valid URI
- Numeric fields are positive

## Future Configuration Files

As the pipeline develops, additional configuration files will be added:

### preprocessing.json (Planned)

Will define image preprocessing parameters:
- Target resolutions
- Color space conversions
- Resize algorithms
- Normalization methods

### encoding.json (Planned)

Will define encoding parameters for each format:
- JPEG: quality levels, chroma subsampling
- WebP: quality, compression method, near-lossless
- AVIF: quality, speed, chroma subsampling
- JPEG XL: effort, distance, modular mode

### quality.json (Planned)

Will define quality measurement settings:
- Which metrics to compute (SSIMULACRA2, Butteraugli, PSNR, SSIM)
- Metric-specific parameters
- Output format (JSON, CSV)

### analysis.json (Planned)

Will define analysis parameters:
- Plot types to generate
- Statistical tests to perform
- Report formats
- Visualization settings

## Loading Configuration

### From Python

```python
from pathlib import Path
from src.dataset import DatasetFetcher

# Uses config/datasets.json by default
fetcher = DatasetFetcher(Path("data/datasets"))

# Or specify custom config
fetcher = DatasetFetcher(
    Path("data/datasets"),
    config_file=Path("custom/datasets.json")
)
```

### From Command Line

```bash
# Uses config/datasets.json by default
python scripts/fetch_dataset.py --list

# Or specify custom config
python scripts/fetch_dataset.py --list --config custom/datasets.json
```

## Configuration Best Practices

1. **Version Control**: Always commit configuration changes
2. **Documentation**: Add comments in commit messages explaining parameter choices
3. **Validation**: Run `--list` or similar commands to validate before committing
4. **Consistency**: Use consistent naming conventions across configs
5. **Comments**: JSON doesn't support comments, so document choices in git commits or related docs
