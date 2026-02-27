---
title: "Fetch datasets"
description: "Download image datasets for use in encoding studies."
---

## Fetch a dataset

Use the dataset ID to download and extract a dataset:

```bash
just fetch div2k-valid
```

The dataset is downloaded, extracted, and stored under `data/datasets/`.

## Available datasets

| ID | Images | Resolution | Format | Size |
|----|--------|------------|--------|------|
| `div2k-valid` | 100 | 2K | PNG | ~449 MB |
| `div2k-train` | 800 | 2K | PNG | ~3.5 GB |
| `liu4k-v1-valid` | 80 | 4K | PNG | ~1.3 GB |
| `liu4k-v1-train` | 800 | 4K | PNG | ~10 GB |
| `liu4k-v2-valid` | 400 | 4K–6K | PNG | ~15 GB |
| `liu4k-v2-train` | 1600 | 4K–6K | PNG | ~60 GB |
| `uhd-iqa-full` | 6073 | 4K | JPEG | ~10.7 GB |

You can also list available datasets with the CLI script:

```bash
python3 scripts/fetch_dataset.py --list
```

## Recommendations

- **Development and testing**: `div2k-valid` (smallest, fast to download)
- **4K research**: `liu4k-v1-valid` (high resolution, manageable size)
- **Large-scale studies**: `liu4k-v2-train` or `uhd-iqa-full`

For unbiased format comparison, prefer lossless PNG datasets (DIV2K, LIU4K)
over UHD-IQA (JPEG source with pre-existing compression artifacts).

## Check downloaded datasets

```bash
python3 scripts/fetch_dataset.py --show-downloaded
```

## Advanced options

The `fetch_dataset.py` script offers additional flags:

```bash
# Keep the archive after extraction (deleted by default)
python3 scripts/fetch_dataset.py div2k-train --keep-archive

# Use a custom datasets directory
python3 scripts/fetch_dataset.py div2k-valid --datasets-dir /path/to/datasets

# Use a custom configuration file
python3 scripts/fetch_dataset.py div2k-valid --config /path/to/datasets.json
```

## Notes on LIU4K datasets

- All LIU4K datasets use **CC BY-NC-ND 4.0** license
- Downloaded from Google Drive (may encounter quota limits)
- LIU4K v2 uses multi-part zip archives and requires **7z** (pre-installed in the dev container)

## Troubleshooting

- **Download fails**: Retry — Google Drive has occasional quota limits. As a fallback, download manually and place files in `data/datasets/`
- **Dataset not found**: Run `python3 scripts/fetch_dataset.py --list` to check available IDs
- **Insufficient space**: Check disk usage; a full pipeline run on DIV2K validation needs ~1 GB free

## See also

- [Datasets reference](../reference/datasets) — detailed dataset properties, licensing, and comparisons
- [Configuration reference](../reference/configuration) — `config/datasets.json` schema
