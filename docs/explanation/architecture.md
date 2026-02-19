---
title: "Project architecture and design"
description: "Overview of the project's architecture, core components, and rationale behind key design decisions."
---

## Pipeline Architecture

The project supports two pipeline workflows:

### Unified Pipeline (Recommended)

The unified pipeline processes images atomically with time-budget control:

```text
For each image (until time budget exhausted):
  └─→ Encode (all variants in parallel)
      └─→ Measure (all variants in parallel)
          └─→ Save results
```

**Key features:**

- **Time-budget control**: Set runtime budget instead of guessing `max_images`
- **Atomic processing**: Each image completes encode+measure before moving to next
- **Memory-backed storage**: Uses `/dev/shm` for temporary files (reduced disk I/O)
- **Error isolation**: Failures on one image don't block others
- **Progress guarantees**: Partial results are always complete and usable

This is the primary workflow for running studies. See `src/pipeline.py` and `scripts/run_pipeline.py`.

### Legacy Separate Pipeline

The traditional linear pipeline separates stages:

```text
Dataset Fetching → Preprocessing → Encoding → Quality Measurement → Analysis
```

Each stage is implemented as an independent module that can be used
standalone or composed into a pipeline. This makes individual components
easy to test and replace.

**Use cases for separate pipeline:**

- Inspecting encoded files before measurement
- Running measurement with different settings
- Debugging encoding issues independently
- Compatibility with existing workflows

See `scripts/encode_images.py` and `scripts/measure_quality.py` for the separate workflow.

### Which Pipeline to Use?

- **Use unified pipeline** for production research runs with predictable runtime
- **Use separate pipeline** for debugging, inspection, or custom measurement workflows

### Data Organization

All research data is organized under the `data/` directory:

```text
data/
├── datasets/        # Raw downloaded datasets
├── preprocessed/    # Preprocessed images
├── encoded/         # Encoded images (JPEG, WebP, AVIF, JXL)
├── metrics/         # Quality measurements (JSON/CSV)
└── analysis/        # Analysis outputs (plots, reports)
```

This structure supports the full pipeline workflow and makes it easy to:

- Track data lineage from raw to analysis
- Organize outputs from each stage
- Share intermediate results
- Clean up specific pipeline stages

### Configuration Management

All pipeline configuration is centralized in the `config/` directory:

- `datasets.json` - Dataset definitions and sources
- Future: `preprocessing.json`, `encoding.json`, `quality.json`, `analysis.json`

This separation allows:

- Version control of configurations
- Easy parameter tuning
- Schema validation
- Configuration sharing between team members

### Dataset Module

The dataset fetching module provides:

- **Extensible architecture** — Easy to add support for new dataset sources
- **DIV2K support** — Training (800 images) and validation (100 images) sets
- **Archive handling** — Automatic download, extraction, and cleanup
- **Progress tracking** — Visual feedback for long-running downloads

The module is designed to support future additions like Flickr2K, HuggingFace
datasets, and custom URL sources without major refactoring.

## Why These Formats?

- **JPEG** — The baseline. Universally supported, well-understood.
- **WebP** — Google's format with broad browser support and good compression.
- **AVIF** — Based on AV1 video codec. Excellent compression at low bitrates.
  The primary focus of this research.
- **JPEG XL** — Next-generation format with advanced features.
  Limited browser support but strong technical merits.

## Why These Quality Metrics?

The project uses both **perceptual** and **traditional** metrics to give
a complete picture:

- **SSIMULACRA2** — Modern perceptual metric designed specifically for
  evaluating lossy image compression. More accurate than SSIM for this use case.
- **Butteraugli** — Models the human visual system to compute perceptual distance.
  Complements SSIMULACRA2 with a different mathematical approach.
- **PSNR** — Simple pixel-difference metric. Not perceptually accurate,
  but widely used and easy to compare with published literature.
- **SSIM** — Structural similarity. Better than PSNR but still not
  specifically designed for compression evaluation.

## Why a Dev Container?

The encoding tools (libaom, dav1d, libjxl, ssimulacra2) require specific
versions built from source. A dev container ensures:

- **Reproducibility** — Everyone gets the exact same tool versions.
- **No host pollution** — Build dependencies don't touch the host system.
- **CI parity** — The same image runs in CI and locally.

## Why Python 3.13?

The project targets a single Python version (3.13) because:

- This is a research project, not a distributable library.
- A single target simplifies testing and avoids compatibility workarounds.
- Python 3.13 is the latest stable release with broad library support.

## Dependency Management

Dependencies are managed entirely through `pyproject.toml`:

- `pip install -e .` — Install production dependencies only.
- `pip install -e ".[dev]"` — Install production + development dependencies
  (pytest, mypy, ruff, type stubs).

There is no separate `requirements.txt`. The `pyproject.toml` is the
single source of truth for all dependencies.

## CI Design

The CI pipeline is structured to be both fast and correct:

1. **Lint & Type Check** runs on bare Ubuntu with Python 3.13 for fast feedback.
2. **Build Image** builds the dev container and pushes to GHCR with layer caching.
3. **Test Suite** runs inside the built container (depends on both steps above),
   ensuring tests execute in the same environment as local development.
4. **Markdown Lint** runs independently in parallel.

Running tests inside the dev container (rather than on bare Ubuntu)
ensures that all encoding and measurement tools are available, so
integration tests produce accurate coverage results.
