---
title: "Architecture and design decisions"
description: "Why the project is structured the way it is — pipeline design, tool choices, and key trade-offs."
---

## Pipeline design

The pipeline processes images **one at a time**, completing all encoding variants and quality measurements for each image before moving to the next:

```text
For each image (until time budget exhausted):
  └─ Encode all variants (formats × quality levels)
      └─ Measure all variants (SSIMULACRA2, Butteraugli, PSNR, SSIM)
          └─ Save results to disk
```

This "atomic per-image" design was chosen over the alternative of encoding all images first, then measuring all:

- **Partial results are always usable.** If a study is interrupted or the time budget expires, every completed image has full encoding + measurement data. There are no orphaned encodings without metrics.
- **Error isolation.** A failure on one image (e.g., an encoder crash on a specific input) does not block other images.

### Time-budget approach

Studies are configured with a **time budget** (e.g., `30m`, `2h`) rather than a fixed image count. The pipeline encodes images from the dataset until the budget expires, then stops. This is more practical than guessing how many images to process:

- Different encoder/quality combinations have wildly different speeds (AVIF speed 0 is ~100× slower than speed 10).
- Multi-format studies multiply the per-image time by the number of variants.
- The user sets a wall-clock time they're willing to wait, and gets as many data points as fit.

### Worker model

Each image is processed by a single worker thread. The `--workers` flag controls parallelism. Because the encoding tools (`cjpeg`, `cwebp`, `avifenc`, `cjxl`) are CPU-intensive native binaries, the bottleneck is CPU time, and Python's GIL does not limit throughput here — subprocess calls release the GIL.

## Study system

A **study** is the central unit of work. Each study is defined by a JSON config file in `config/studies/` that specifies:

- Which dataset to use
- Which formats and parameter ranges to encode
- Study metadata (name, description)

The study ID (filename without `.json`) is used everywhere: directory names under `data/encoded/<study-id>/`, `data/metrics/<study-id>/`, `data/analysis/<study-id>/`, and in CLI commands (`just pipeline <study-id> <time-budget>`).

This design means:

- **Adding a new experiment** is just creating a new JSON file — no code changes.
- **Reproducibility** — the config file fully describes what was run.
- **Multiple studies coexist** — each study's outputs are isolated in its own subdirectory.

## Data separation

All data lives under `data/`, strictly separated from code:

```text
data/
├── datasets/        # Raw downloads (from just fetch)
├── preprocessed/    # Resolution-scaled images (per study)
├── encoded/         # Compressed images (per study, per format)
├── metrics/         # Quality measurements (per study)
├── analysis/        # Plots and statistics (per study)
└── report/          # Generated HTML reports
```

Everything under `data/` is git-ignored (except `.gitkeep` markers). This means:

- The repository stays small regardless of how many studies are run.
- Datasets can be re-downloaded; encoded/metrics/analysis can be regenerated.
- Code changes are cleanly separated from data changes in version control.

## Configuration over code

Pipeline parameters live in JSON config files rather than being hardcoded:

- `config/datasets.json` — dataset sources, URLs, storage types
- `config/studies/*.json` — study definitions (formats, quality ranges, speeds)
- JSON schemas (`config/*.schema.json`) — validate configs at load time

There are no "future" config files for preprocessing, quality, or analysis — those behaviours are determined by the study config and the source code defaults. This keeps the config surface small and avoids speculative abstraction.

See [Add a custom dataset](../how-to/add-dataset) and [Create a custom study](../how-to/create-study) for guides on extending the configuration.

## Format choices

| Format | Role | Why included |
|--------|------|-------------|
| JPEG | Baseline | Universal support, well-understood, the format everything is compared against |
| WebP | Established alternative | Broad browser support, good compression, Google-backed |
| AVIF | Primary research target | Based on AV1, excellent low-bitrate compression, the main focus of this project |
| JPEG XL | Next-generation | Strong technical merits (progressive decode, lossless round-trip), limited browser support |

## Metric choices

The project measures both **perceptual** and **traditional** metrics:

| Metric | Type | Why |
|--------|------|-----|
| SSIMULACRA2 | Perceptual | Designed specifically for lossy compression evaluation; most accurate for this use case |
| Butteraugli | Perceptual | Models human visual system with a different mathematical approach; complements SSIMULACRA2 |
| PSNR | Traditional | Simple, widely used, easy to compare with published literature |
| SSIM | Traditional | Better than PSNR for structural comparison, well-known baseline |

Perceptual metrics (SSIMULACRA2, Butteraugli) are prioritised in analysis because they correlate better with human judgement than PSNR/SSIM for compression artifacts.

## Dev container

The encoding tools (`avifenc`, `cjxl`, `cwebp`, `cjpeg`) and measurement tools (`ssimulacra2`, `butteraugli_main`, `avifdec`, `djxl`) require specific builds. A dev container ensures:

- **Reproducibility** — everyone gets the exact same tool versions.
- **No host pollution** — build dependencies don't touch the host system.
- **CI parity** — the same image runs in CI and locally.

## Python 3.13, single target

This is a research project, not a distributable library. Targeting a single Python version (3.13) simplifies testing, avoids compatibility workarounds, and lets the code use the latest language features without conditional logic.

## Dependency management

All dependencies are declared in `pyproject.toml`:

- `pip install -e .` — production dependencies
- `pip install -e ".[dev]"` — adds pytest, mypy, ruff, type stubs

There is no `requirements.txt`. The pyproject is the single source of truth.

## CI design

1. **Lint & type check** — runs on bare Ubuntu with Python 3.13 for fast feedback.
2. **Build image** — builds the dev container and pushes to GHCR with layer caching.
3. **Test suite** — runs inside the built container (depends on both above), ensuring encoding/measurement tools are available for integration tests.
4. **Markdown lint** — runs independently in parallel.

Running tests inside the dev container (rather than on bare Ubuntu) ensures that all native tools are available and integration tests produce accurate results.

Beyond CI, the project also uses GitHub Actions as a public research platform — see [Public research with GitHub Actions](github-actions-research) for details.

## Module architecture

The `src/` modules map to pipeline stages and post-processing:

| Module | Purpose |
|--------|---------|
| `study.py` | Load and validate study configs |
| `dataset.py` | Fetch and manage image datasets |
| `preprocessing.py` | Resize images by longest edge |
| `encoder.py` | Encode images via subprocess calls to native tools |
| `quality.py` | Measure quality metrics via subprocess calls |
| `pipeline.py` | Orchestrate encode → measure per image with time budget |
| `analysis.py` | Generate plots and statistics from quality results |
| `comparison.py` | Generate side-by-side visual comparison images |
| `interactive.py` | Build interactive HTML report |
| `report_images.py` | Generate report visualisation assets |

Each module is independently testable. Scripts in `scripts/` provide CLI entry points that compose these modules.

For guidance on extending the codebase with new formats or metrics, see [Extend formats and metrics](../how-to/extend-formats-and-metrics).
