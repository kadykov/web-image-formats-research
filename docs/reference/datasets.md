---
title: "Datasets"
description: "Properties, licensing, and comparison of supported image datasets."
---

## Comparison table

| Feature | DIV2K | LIU4K v1 | LIU4K v2 | UHD-IQA |
|---------|-------|----------|----------|---------|
| **Resolution** | 2K | 4K | 4K–6K | 4K (3840×2160) |
| **Format** | PNG (lossless) | PNG (lossless) | PNG (lossless) | JPEG (lossy) |
| **Images (val)** | 100 | 80 | 400 | ~2000 |
| **Images (train)** | 800 | 800 | 1600 | ~4000 |
| **Size (val)** | 449 MB | 1.3 GB | 15 GB | ~3.5 GB |
| **Size (train)** | 3.5 GB | 10 GB | 60 GB | ~7 GB |
| **Storage** | Direct HTTP | Google Drive | Google Drive | Direct HTTP |
| **License** | Unknown | CC BY-NC-ND 4.0 | CC BY-NC-ND 4.0 | CC0 |
| **Post-process** | None | None | Extract multi-part zips | None |

## DIV2K (2K)

- **Source**: [ETH Zurich Computer Vision Lab](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Resolution**: 2K (variable aspect ratios)
- **Format**: PNG (lossless)
- **License**: Unknown — check DIV2K website
- **IDs**: `div2k-valid` (100 images, 449 MB), `div2k-train` (800 images, 3.5 GB)

Best for development and testing due to manageable size and direct HTTP download.

## LIU4K v1 (4K)

- **Source**: [Peking University — LIU4K Benchmark](https://github.com/flyywh/Liu4K_benchmark)
- **Resolution**: 4K
- **Format**: PNG (lossless)
- **License**: CC BY-NC-ND 4.0 (attribution required, non-commercial, no derivatives)
- **Storage**: Google Drive (requires `gdown`)
- **IDs**: `liu4k-v1-valid` (80 images, 1.3 GB), `liu4k-v1-train` (800 images, 10 GB)

Recommended for 4K research. Single-file zip archive, simpler than v2.

## LIU4K v2 (4K–6K)

- **Source**: [Peking University — LIU4K v2](https://structpku.github.io/LIU4K_Dataset/LIU4K_v2.html)
- **Resolution**: 4K–6K (minimum 3K)
- **Format**: PNG (lossless)
- **License**: CC BY-NC-ND 4.0
- **Storage**: Google Drive — multi-part zip archives by category
- **IDs**: `liu4k-v2-valid` (400 images, 15 GB), `liu4k-v2-train` (1600 images, 60 GB)
- **Requires**: 7z for multi-part zip extraction (pre-installed in dev container)

Larger and higher resolution than v1. Categories: Animal, Building, Capture, Mountain, Street.

## UHD-IQA (4K JPEG)

- **Source**: [University of Konstanz — UHD-IQA Database](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html)
- **Resolution**: UHD-1 (3840×2160)
- **Format**: JPEG (lossy, sourced from Pixabay)
- **License**: CC0 (Public Domain)
- **ID**: `uhd-iqa-full` (6073 images, 10.7 GB)
- **Includes**: Quality ratings (MOS) and metadata

Largest dataset by image count. However, JPEG source images contain pre-existing
compression artifacts, making this dataset less suitable for unbiased format comparison.
Best for large-scale testing and quality assessment research.

## Recommendations by use case

| Use case | Recommended dataset |
|----------|-------------------|
| Development and testing | `div2k-valid` — fast download, manageable size |
| 4K format research | `liu4k-v1-valid` — lossless 4K, reasonable size |
| Large-scale 4K study | `liu4k-v2-train` — 1600 lossless images up to 6K |
| Scale testing | `uhd-iqa-full` — 6000+ images (note: JPEG source) |
| Unbiased format comparison | DIV2K or LIU4K (lossless PNG, no pre-existing artifacts) |

## Storage location

All datasets are stored under `data/datasets/`:

```text
data/datasets/
├── DIV2K_valid/
├── DIV2K_train/
├── LIU4K_valid/
├── LIU4K_train/
└── UHD_IQA/
```

## See also

- [Fetch datasets how-to](../how-to/fetch-datasets) — download commands and troubleshooting
- [Configuration reference](configuration) — `config/datasets.json` schema
