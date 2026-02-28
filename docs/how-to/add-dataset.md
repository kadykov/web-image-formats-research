---
title: "Add a custom dataset"
description: "Register a new image dataset for use in encoding studies — no code changes required."
---

Adding a dataset only requires editing `config/datasets.json`.
No Python code changes are needed.

## Add a direct-download dataset

Open `config/datasets.json` and add an entry to the `datasets` array:

```json
{
  "id": "my-dataset",
  "name": "My Custom Dataset",
  "description": "A collection of high-resolution test images",
  "type": "zip",
  "url": "https://example.com/my-dataset.zip",
  "size_mb": 200,
  "image_count": 50,
  "resolution": "2K",
  "format": "PNG",
  "extracted_folder": "my-dataset-images",
  "license": "CC BY 4.0",
  "source": "My Organization"
}
```

### Required fields

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (lowercase, hyphens, digits only — pattern `^[a-z0-9-]+$`). Used in CLI commands. |
| `name` | Human-readable name. |
| `description` | Brief description. |
| `type` | Archive format: `zip`, `tar`, `tar.gz`, `tgz`, or `folder` (Google Drive folders). |
| `url` | Download URL (HTTP/HTTPS). |
| `size_mb` | Approximate download size in MB. |
| `image_count` | Number of images. |
| `resolution` | Resolution description (e.g., `"2K"`, `"4K"`). |
| `format` | Image format (e.g., `"PNG"`, `"TIFF"`). |

### Optional fields

| Field | Description |
|-------|-------------|
| `storage_type` | `"direct"` (default), `"google_drive"`, or `"dropbox"`. |
| `folder_id` | Google Drive folder ID (for `"google_drive"` storage type). |
| `post_process` | `"extract_zips"` or `"extract_multipart_zips"` (for nested archives). |
| `extracted_folder` | Folder name inside the archive after extraction. |
| `rename_to` | Rename extracted folder to this name. |
| `license` | License information. |
| `source` | Organization providing the dataset. |

## Add a Google Drive dataset

For datasets hosted on Google Drive, set `storage_type` and provide the
folder or file details:

```json
{
  "id": "my-gdrive-dataset",
  "name": "My Google Drive Dataset",
  "description": "4K images from Google Drive",
  "type": "folder",
  "url": "https://drive.google.com/drive/folders/FOLDER_ID",
  "storage_type": "google_drive",
  "folder_id": "FOLDER_ID",
  "post_process": "extract_zips",
  "size_mb": 1300,
  "image_count": 80,
  "resolution": "4K",
  "format": "PNG",
  "license": "CC BY-NC-ND 4.0",
  "source": "Research Group"
}
```

## Verify the dataset

Fetch the new dataset:

```bash
just fetch my-dataset
```

Check that images are in place:

```bash
ls data/datasets/my-dataset/
```

You can also verify all downloaded datasets:

```bash
python3 scripts/fetch_dataset.py --show-downloaded
```

## Image format recommendations

- **Use lossless source images** (PNG, TIFF) for unbiased format comparison.
  JPEG sources have pre-existing compression artifacts that skew quality metrics.
- **Include at least 50 images** for statistical significance in analysis.
- **Consistent resolution** within a dataset makes comparison cleaner,
  though the `resolution` encoder parameter can normalize images before encoding.

## Schema validation

The configuration is validated against `config/datasets.schema.json`.
IDEs with JSON Schema support (including VS Code in the dev container)
will provide autocompletion and inline validation when you edit the file.

## See also

- [Datasets reference](../reference/datasets) — properties and licensing for built-in datasets
- [Configuration reference](../reference/configuration) — full `datasets.json` schema
- [Fetch datasets](fetch-datasets) — download commands and troubleshooting
