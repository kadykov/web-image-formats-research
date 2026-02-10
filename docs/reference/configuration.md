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

## Study Configuration Files

Study configurations live in the `studies/` directory
and define targeted encoding experiments.

### Schema

Validated by `config/study.schema.json`.

### Structure

```json
{
  "id": "string",
  "name": "string",
  "description": "string (optional)",
  "dataset": {
    "id": "string",
    "max_images": "integer (optional)"
  },
  "preprocessing": {
    "resize": [1920, 1280, 640]
  },
  "encoders": [
    {
      "format": "jpeg|webp|avif|jxl",
      "quality": 75,
      "chroma_subsampling": ["444", "420"],
      "speed": [4, 6],
      "extra_args": {}
    }
  ]
}
```

### Quality Specification

The `quality` field supports three formats:

| Format | Example | Result |
|--------|---------|--------|
| Single integer | `75` | `[75]` |
| Explicit list | `[60, 75, 90]` | `[60, 75, 90]` |
| Range object | `{"start": 30, "stop": 90, "step": 10}` | `[30, 40, 50, 60, 70, 80, 90]` |

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the study. |
| `name` | string | No | Human-readable name. Defaults to `id`. |
| `description` | string | No | Purpose of the study. |
| `dataset.id` | string | Yes | Dataset identifier from `datasets.json`. |
| `dataset.max_images` | integer | No | Limit images from dataset (useful for testing). |
| `preprocessing.resize` | integer[] | No | Target resolutions (longest edge in pixels). |
| `encoders[].format` | string | Yes | One of: `jpeg`, `webp`, `avif`, `jxl`. |
| `encoders[].quality` | int/list/range | Yes | Quality settings to sweep (0–100). |
| `encoders[].chroma_subsampling` | string[] | No | Subsampling modes: `444`, `422`, `420`, `400`. |
| `encoders[].speed` | int/int[] | No | Encoder speed/effort setting(s). |
| `encoders[].extra_args` | object | No | Encoder-specific CLI arguments. |

### Example: Quality Sweep

```json
{
  "id": "avif-quality-sweep",
  "name": "AVIF Quality Sweep",
  "dataset": { "id": "div2k-valid", "max_images": 10 },
  "encoders": [
    {
      "format": "avif",
      "quality": { "start": 30, "stop": 90, "step": 5 },
      "chroma_subsampling": ["444", "420"],
      "speed": 4
    }
  ]
}
```

### Example: Format Comparison

```json
{
  "id": "format-comparison",
  "name": "Format Comparison",
  "dataset": { "id": "div2k-valid", "max_images": 10 },
  "encoders": [
    { "format": "jpeg", "quality": [60, 75, 85, 95] },
    { "format": "webp", "quality": [60, 75, 85, 95] },
    { "format": "avif", "quality": [60, 75, 85, 95], "speed": 4 },
    { "format": "jxl",  "quality": [60, 75, 85, 95] }
  ]
}
```

### Running Studies

```bash
# List available studies
just list-studies

# Preview what a study will do
just dry-run-study studies/avif-quality-sweep.json

# Run a study
just run-study studies/avif-quality-sweep.json
```

## Encoding Results

The output of each study run is a JSON file matching
`config/encoding-results.schema.json`.

### Structure

```json
{
  "study_id": "string",
  "study_name": "string",
  "dataset": {
    "id": "string",
    "path": "string",
    "image_count": "integer"
  },
  "timestamp": "ISO 8601 string",
  "encodings": [
    {
      "source_image": "string",
      "original_image": "string",
      "encoded_path": "string",
      "format": "string",
      "quality": "integer",
      "file_size": "integer",
      "width": "integer",
      "height": "integer",
      "source_file_size": "integer",
      "chroma_subsampling": "string (optional)",
      "speed": "integer (optional)",
      "resolution": "integer (optional)",
      "extra_args": "object (optional)"
    }
  ]
}
```

The results file is saved at `data/encoded/<study-id>/results.json`
and serves as the input for the quality measurement stage.

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
