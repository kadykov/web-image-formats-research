---
title: "Code modules reference"
description: "Overview of Python modules in `src/`, their responsibilities, and key public APIs for integration."
---

## Pipeline Overview

```text
Dataset Fetching → Preprocessing → Encoding → Quality Measurement → Analysis
```

## `src/dataset.py` — Dataset Fetching

**Class: `DatasetFetcher`**

Downloads and organizes test image datasets.

- `__init__(base_dir: Path)` — Set base directory for datasets
- `download_image(url: str, output_path: Path) -> bool` — Download a single image
- `list_datasets() -> list[str]` — List available dataset directories

## `src/preprocessing.py` — Image Preprocessing

**Class: `ImagePreprocessor`**

Prepares images for encoding.

- `__init__(output_dir: Path)` — Set output directory
- `resize_image(input_path, target_size, ...) -> Path` — Resize with optional aspect ratio preservation
- `convert_to_png(input_path, ...) -> Path` — Convert any image to PNG
- `get_image_info(image_path) -> dict` — Extract format, dimensions, mode, file size

## `src/encoder.py` — Image Encoding

**Class: `ImageEncoder`**

Encodes images to various formats using external CLI tools.

- `__init__(output_dir: Path)` — Set output directory
- `encode_jpeg(input_path, quality, ...) -> EncodeResult` — Encode via `cjpeg`
- `encode_webp(input_path, quality, ...) -> EncodeResult` — Encode via `cwebp`
- `encode_avif(input_path, quality, speed=4, ...) -> EncodeResult` — Encode via `avifenc`
- `encode_jxl(input_path, quality, ...) -> EncodeResult` — Encode via `cjxl`

**Dataclass: `EncodeResult`**

- `success: bool`
- `output_path: Path | None`
- `file_size: int | None`
- `error_message: str | None`

## `src/quality.py` — Quality Measurement

**Class: `QualityMeasurer`**

Measures image quality using external tools.

- `measure_ssimulacra2(original, compressed) -> float | None`
- `measure_psnr(original, compressed) -> float | None`
- `measure_ssim(original, compressed) -> float | None`
- `measure_butteraugli(original, compressed) -> float | None`
- `measure_all(original, compressed) -> QualityMetrics`

**Dataclass: `QualityMetrics`**

- `ssimulacra2: float | None`
- `psnr: float | None`
- `ssim: float | None`
- `butteraugli: float | None`
- `error_message: str | None`

## `src/analysis.py` — Analysis and Visualization

**Class: `CompressionAnalyzer`**

Analyzes compression results and generates plots.

- `__init__(results_dir: Path)` — Set results directory
- `create_dataframe(results) -> DataFrame` — Build DataFrame from result dicts
- `calculate_compression_ratio(df) -> DataFrame` — Add ratio and percentage columns
- `plot_quality_vs_size(df, quality_metric, ...) -> None` — Scatter plot with trend lines
- `plot_compression_efficiency(df, ...) -> None` — Bar chart comparing formats
- `generate_summary_report(df, ...) -> str` — Markdown summary report
