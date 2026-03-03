---
title: "Code modules"
description: "Overview of Python modules in `src/` and their responsibilities."
---

This is a lightweight overview. For full API signatures and docstrings,
see the auto-generated [API reference](api/).

## Pipeline overview

```text
StudyConfig → PipelineRunner → Analysis → Comparison → Report
```

## `src/study.py` — Study configuration

Loads and validates study JSON configs from `config/studies/`.

- `StudyConfig` — dataclass with id, name, dataset_id, encoders, time_budget, description
- `EncoderConfig` — per-format settings: quality, chroma_subsampling, speed, effort, method, resolution

## `src/dataset.py` — Dataset fetching

Downloads and manages image datasets using `config/datasets.json`.

- `DatasetFetcher` — main class: `fetch_dataset()`, `list_available_datasets()`, `get_dataset_config()`
- `DatasetConfig` — dataclass with id, name, url, storage_type, folder_id, post_process, etc.
- Supports direct HTTP, Google Drive, and Dropbox downloads

## `src/preprocessing.py` — Image preprocessing

Resizes images for resolution-impact studies.

- `ImagePreprocessor` — `resize_image()` with configurable target size and aspect ratio

## `src/encoder.py` — Image encoding

Encodes images using external CLI tools (cjpeg, cwebp, avifenc, cjxl).

- `ImageEncoder` — `encode_jpeg()`, `encode_webp()`, `encode_avif()`, `encode_jxl()`
- `EncodeResult` — dataclass: success, output_path, file_size, error_message
- `get_encoder_version()` — query version of installed encoders
- All encoders forced to single-threaded mode for consistent benchmarking

## `src/quality.py` — Quality measurement

Measures perceptual and traditional quality metrics.

- `QualityMeasurer` — `measure_ssimulacra2()`, `measure_psnr()`, `measure_ssim()`, `measure_butteraugli()`, `measure_butteraugli_with_distmap()`, `measure_all()`
- `QualityMetrics` — dataclass: ssimulacra2, psnr, ssim, butteraugli
- `QualityRecord`, `QualityResults` — structured containers for pipeline output, with `save()` to JSON
- `to_png()` — converts AVIF/JXL/WebP to PNG for measurement tools
- `read_pfm()` — reads Butteraugli PFM distortion maps
- `find_worst_region_in_array()` — sliding-window worst-region detection using integral images

## `src/pipeline.py` — Unified pipeline

Runs encode + measure for each image with time-budget control.

- `PipelineRunner` — main class: `run(config, time_budget, save_artifacts, save_worst_image, num_workers)`
- `parse_time_budget()` — parses "30m", "1h", "90s" format strings
- Worker-per-image architecture using `ProcessPoolExecutor`

## `src/analysis.py` — Statistical analysis

Computes statistics and generates static matplotlib plots.

- `load_quality_results()` — load quality.json
- `create_analysis_dataframe()` — build DataFrame with derived metrics (bytes_per_pixel, compression_ratio, efficiency)
- `compute_statistics()` — mean, min, max, percentiles (5/25/50/75/95)
- `determine_sweep_parameter()`, `determine_secondary_sweep_parameter()` — auto-detect what varies
- `plot_quality_metrics()`, `plot_rate_distortion()` — SVG output with mean + worst-case lines
- `analyze_study()` — full analysis pipeline entry point

## `src/interactive.py` — Interactive visualizations

Generates interactive Plotly figures for the HTML report.

- `generate_study_figures()` — creates all Plotly figures for a study
- `plot_quality_vs_param()`, `plot_rate_distortion()` — Plotly equivalents of analysis plots
- `figure_to_html_fragment()` — convert Plotly figure to embeddable HTML div

## `src/comparison.py` — Visual comparisons

Generates side-by-side comparison images with distortion maps.

- `ComparisonConfig` — crop_size, zoom_factor, tile_parameter, source_image, study_config_path
- `select_best_image()` — select most representative image by cross-format CV (std / mean)
- `generate_distortion_map()` — create Butteraugli PFM distortion maps
- `generate_comparison()` — main entry point: select image, crop fragment by std-dev map, assemble montage
- Uses ImageMagick 7 (`magick montage`) for grid assembly

## `src/report_images.py` — Report image optimization

Optimizes comparison images for the HTML report with responsive formats.

- `optimise_lossless()` — single lossless WebP variant
- `optimise_lossy()` — multi-format (AVIF + WebP) at multiple widths
- `discover_and_optimise()` — find and optimize all comparison images for a study
- `picture_html()`, `img_srcset_html()` — generate `<picture>` and `<img srcset>` HTML
