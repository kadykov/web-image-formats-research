---
title: "Configuration reference"
description: "Schema and field reference for dataset and study configuration files."
---

## `config/datasets.json`

Defines all datasets available for fetching. Validated by `config/datasets.schema.json`.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (`^[a-z0-9-]+$`). Used in CLI commands. |
| `name` | string | Yes | Human-readable name. |
| `description` | string | Yes | Brief description. |
| `type` | string | Yes | Archive format: `zip`, `tar`, `tar.gz`, `tgz`, or `folder` (Google Drive). |
| `url` | string | Yes | Download URL (HTTP/HTTPS, Google Drive, Dropbox). |
| `size_mb` | number | Yes | Approximate download size in MB. |
| `image_count` | number | Yes | Number of images. |
| `resolution` | string | Yes | Resolution description (e.g., "2K", "4K"). |
| `format` | string | Yes | Image format (e.g., "PNG", "JPEG"). |
| `storage_type` | string | No | Storage provider: `direct` (default), `google_drive`, or `dropbox`. |
| `folder_id` | string | No | Google Drive folder ID (for folder-type downloads). |
| `post_process` | string | No | Post-processing action: `extract_multipart_zips` (for LIU4K v2). |
| `extracted_folder` | string | No | Folder name after extraction. |
| `rename_to` | string | No | Rename extracted folder to this name. |
| `license` | string | No | License information. |
| `source` | string | No | Organization providing the dataset. |

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

## Study configuration files

Study configs live in `config/studies/` and define encoding experiments.
Validated by `config/study.schema.json`.

### Available studies

| File | Study ID | Description |
|------|----------|-------------|
| `format-comparison.json` | `format-comparison` | Compare JPEG, WebP, AVIF, JPEG XL |
| `avif-speed-sweep.json` | `avif-speed-sweep` | AVIF speed parameter sweep |
| `avif-chroma-subsampling.json` | `avif-chroma-subsampling` | AVIF chroma subsampling comparison |
| `jxl-effort-sweep.json` | `jxl-effort-sweep` | JPEG XL effort level comparison |
| `webp-method-sweep.json` | `webp-method-sweep` | WebP method parameter sweep |
| `resolution-impact.json` | `resolution-impact` | Impact of resolution on quality |

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique study identifier. |
| `name` | string | No | Human-readable name. Defaults to `id`. |
| `description` | string | No | Purpose of the study. |
| `time_budget` | number | No | Default time budget in seconds. Overridden by CLI. |
| `dataset.id` | string | Yes | Dataset identifier from `datasets.json`. |
| `dataset.max_images` | integer | No | Limit images from dataset. |
| `encoders` | array | Yes | List of encoder configurations. |

### Encoder fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format` | string | Yes | One of: `jpeg`, `webp`, `avif`, `jxl`. |
| `quality` | int/list/range | Yes | Quality settings to sweep (0–100). |
| `chroma_subsampling` | string[] | No | Subsampling modes: `444`, `422`, `420`, `400`. |
| `speed` | int/int[] | No | AVIF speed setting(s) (0–10). |
| `effort` | int/int[] | No | JXL effort setting(s) (1–10). |
| `method` | int/int[] | No | WebP method setting(s) (0–6). |
| `resolution` | int/int[] | No | Target resolution(s) in pixels (longest edge). |
| `extra_args` | object | No | Encoder-specific CLI arguments. |

### Quality specification formats

| Format | Example | Expanded |
|--------|---------|----------|
| Single integer | `75` | `[75]` |
| Explicit list | `[60, 75, 90]` | `[60, 75, 90]` |
| Range object | `{"start": 30, "stop": 90, "step": 10}` | `[30, 40, 50, 60, 70, 80, 90]` |

### Example: format comparison

```json
{
  "id": "format-comparison",
  "name": "Format Comparison",
  "time_budget": 1800,
  "dataset": { "id": "div2k-valid", "max_images": 10 },
  "encoders": [
    { "format": "jpeg", "quality": [60, 75, 85, 95] },
    { "format": "webp", "quality": [60, 75, 85, 95] },
    { "format": "avif", "quality": [60, 75, 85, 95], "speed": 4 },
    { "format": "jxl",  "quality": [60, 75, 85, 95] }
  ]
}
```

### Example: parameter sweep with range

```json
{
  "id": "avif-speed-sweep",
  "name": "AVIF Speed Sweep",
  "time_budget": 3600,
  "dataset": { "id": "div2k-valid", "max_images": 10 },
  "encoders": [
    {
      "format": "avif",
      "quality": { "start": 30, "stop": 90, "step": 10 },
      "speed": [2, 4, 6, 8]
    }
  ]
}
```

## Quality results schema

Pipeline output is saved as `data/metrics/<study-id>/quality.json`,
validated by `config/quality-results.schema.json`.

Key fields per measurement:

| Field | Description |
|-------|-------------|
| `source_image` | Path to preprocessed source image |
| `original_image` | Path to original dataset image |
| `format` | Encoding format |
| `quality` | Quality setting used |
| `file_size` | Encoded file size in bytes |
| `width`, `height` | Image dimensions |
| `ssimulacra2` | SSIMULACRA2 score (or null on error) |
| `psnr` | PSNR in dB |
| `ssim` | SSIM score |
| `butteraugli` | Butteraugli distance |
| `encoding_time` | Time taken to encode in seconds |
| `measurement_error` | Error message (null if successful) |
