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

- `StudyConfig` — dataclass with study metadata plus analysis/comparison settings and crop-impact options
- `EncoderConfig` — per-format settings: quality, chroma_subsampling, speed, effort, method, resolution, crop

## `src/dataset.py` — Dataset fetching

Downloads and manages image datasets using `config/datasets.json`.

- `DatasetFetcher` — main class: `fetch_dataset()`, `list_available_datasets()`, `get_dataset_config()`
- `DatasetConfig` — dataclass with id, name, url, storage_type, folder_id, post_process, etc.
- Supports direct HTTP, Google Drive, and Dropbox downloads

## `src/preprocessing.py` — Image preprocessing

Resizes and crops images for preprocessing-driven studies.

- `ImagePreprocessor` — `resize_image()` and `crop_image_around_fragment()`
- `CropResult` — cropped image path plus the crop rectangle in original-image coordinates

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
- `QualityRecord`, `QualityResults` — structured containers for pipeline output, including crop metadata, with `save()` to JSON
- `to_png()` — converts AVIF/JXL/WebP to PNG for measurement tools
- `read_pfm()` — reads Butteraugli PFM distortion maps
- `find_worst_region_in_array()` — sliding-window worst-region detection using integral images

## `src/pipeline.py` — Unified pipeline

Runs encode + measure for each image with time-budget control.

- `PipelineRunner` — main class: `run(config, time_budget, save_artifacts, save_worst_image, num_workers)`
- `parse_time_budget()` — parses "30m", "1h", "90s" format strings
- Worker-per-image architecture using `ProcessPoolExecutor`
- Supports resolution sweeps and crop-impact studies with fragment-only metric measurement

## `src/analysis.py` — Statistical analysis

Computes statistics and generates static matplotlib plots.

- `load_quality_results()` — load quality.json
- `create_analysis_dataframe()` — build DataFrame with derived metrics (bits_per_pixel, compression_ratio, efficiency)
- `compute_statistics()` — mean, min, max, percentiles (5/25/50/75/95)
- `determine_sweep_parameter()`, `determine_secondary_sweep_parameter()` — auto-detect what varies
- `plot_quality_metrics()`, `plot_rate_distortion()` — SVG output with mean + worst-case lines
- `analyze_study()` — full analysis pipeline entry point

## `src/interactive.py` — Interactive visualizations

Generates interactive Plotly figures for the HTML report.

- `generate_study_figures()` — creates all Plotly figures for a study
- `plot_quality_vs_param()`, `plot_rate_distortion()` — Plotly equivalents of analysis plots
- `figure_to_html_fragment()` — convert Plotly figure to embeddable HTML div

## `src/interpolation.py` — Quality interpolation

Estimates encoder quality settings that produce a desired output metric value,
and selects the most representative source image for comparison figures.

- `interpolate_quality_for_metric()` — find encoder quality for a target metric value (linear or cubic-spline)
- `interpolate_metric_at_quality()` — predict output metric at a given quality setting
- `compute_cross_format_cv()` — coefficient of variation (std / mean) of an output metric across tile-parameter values for one image
- `select_best_image()` — pick the source image with the highest mean CV across target values
- `_collect_quality_metric_pairs()` — filter and sort (quality, metric) pairs from measurement data

## `src/comparison.py` — Visual comparisons

Generates side-by-side comparison figures via interpolation-based quality matching.
Fully decoupled from the pipeline: re-encodes images on the fly using quality
settings interpolated from `quality.json`.

- `ComparisonConfig` — crop_size, zoom_factor, max_columns, distmap_vmax, tile_parameter, source_image, study_config_path
- `generate_comparison()` — main entry point: select image by CV, interpolate quality, re-encode, build aggregate std map, crop fragment, assemble grids
- `generate_distortion_map()` — create raw Butteraugli PFM distortion maps
- `find_worst_region()` — sliding-window worst-region search on a PFM map
- `crop_and_zoom()` — crop a fragment and scale with nearest-neighbour interpolation
- `assemble_comparison_grid()` — assemble labeled tiles with ImageMagick 7 `montage`, including fixed-position placeholders for missing variants
- `encode_image()` — re-encode a source image using measurement-record parameters
- `_anisotropic_std_map()` — compute aggregate per-pixel std map across distortion maps

## `src/report_images.py` — Report image optimization

Optimizes comparison images for the HTML report with responsive formats.

- `optimise_lossless()` — single lossless WebP variant
- `optimise_lossy()` — multi-format (AVIF + WebP) at multiple widths
- `discover_and_optimise()` — find and optimize all comparison images for a study
- `picture_html()`, `img_srcset_html()` — generate `<picture>` and `<img srcset>` HTML
