---
title: "Public research with GitHub Actions"
description: "How this project uses GitHub Actions as a free, transparent, and reproducible research platform."
---

## The idea

This project uses GitHub Actions not just for CI/CD, but as a **public
research platform**. Encoding studies run on GitHub-hosted runners, and
the results — including every parameter, tool version, and measurement —
are published openly through GitHub Releases and GitHub Pages.

Because this is an open-source repository, every workflow run is visible:
anyone can inspect the exact execution logs, configuration inputs, and
timing of each study. This transparency is the foundation of reproducible
image format research.

## Why GitHub Actions

### Free infrastructure for open source

GitHub provides generous free-tier runners for public repositories:

| Resource | Specification |
|----------|---------------|
| CPU | 4 cores (x86_64) |
| RAM | 16 GB |
| Storage | 14 GB SSD |
| Job timeout | 6 hours |
| Concurrent jobs | 20 |

This is sufficient to run studies on the DIV2K validation dataset
(100 images, ~450 MB) with multiple format and parameter combinations
within a single workflow run.

### Transparent execution

Every workflow run records:

- **Input parameters** — time budget, study selection
- **Tool versions** — encoder and metric tool versions are logged and
  saved in quality results JSON
- **Execution logs** — full stdout/stderr for every step
- **Timing** — per-step durations visible in the Actions UI

Anyone can verify what was run and how by examining the workflow run page.

### Deterministic encoding

The project enforces single-threaded encoding (`avifenc -j 1`,
`cjxl --num_threads=1`) so that results are deterministic regardless
of the runner's CPU count. Combined with pinned tool versions in the
Dockerfile (e.g., libjxl v0.11.2, libavif v1.3.0, libaom v3.13.1),
this ensures that running the same study twice produces identical
encoded files and quality measurements.

## Workflow architecture

The study workflow (`.github/workflows/study.yml`) is triggered manually
via `workflow_dispatch` and orchestrates seven jobs:

```text
build-image ──┐
              ├── fetch-dataset ──┐
prepare ──────┘                   │
                                  ├── run-study (parallel per study)
                                  │      ├── pipeline (encode + measure)
                                  │      ├── analyze (statistics + plots)
                                  │      └── compare (visual comparisons)
                                  │
                                  └── generate-report
                                         ├── interactive HTML report
                                         ├── release notes
                                         └── release assets (CSV)
                                              │
                                         deploy-report → GitHub Pages
                                         release → GitHub Release
```

### Key design decisions

- **Dev container image**: The workflow builds the same dev container used
  for local development and pushes it to GHCR. Every subsequent job runs
  inside this image, ensuring tool parity between CI and local environments.

- **Parallel studies**: Each study runs as an independent matrix job.
  A format-comparison study and an AVIF speed sweep run simultaneously on
  separate runners, maximizing throughput within the 6-hour window.

- **Artifact pipeline**: Study results flow through GitHub Actions artifacts.
  The dataset is fetched once and shared. Each study uploads its metrics and
  analysis. The report job downloads all results and generates the final
  report.

## Published outputs

### Interactive report (GitHub Pages)

The report is deployed to the `report/` subdirectory of GitHub Pages.
It includes interactive Plotly visualizations for rate-distortion curves,
quality-vs-parameter plots, and visual comparisons with Butteraugli
distortion maps.

### GitHub Releases

Each workflow run creates a timestamped release
(e.g., `study-20260228-143000`) containing:

- **Release notes** — auto-generated markdown summarizing studies,
  datasets, tool versions, and key findings
- **CSV statistics files** — per-study statistical summaries suitable
  for independent re-analysis in any tool (Excel, R, Python, etc.)

The CSV files include per-format, per-quality-level aggregated statistics
(mean, median, percentiles) for all quality metrics, file sizes, encoding
times, and derived efficiency metrics.

### Data availability

All data from the publicly performed research is available for re-analysis:

| Output | Location | Retention |
|--------|----------|-----------|
| Interactive report | GitHub Pages `/report/` | Until next deployment |
| CSV statistics | GitHub Releases | Permanent |
| Raw metrics JSON | Workflow artifacts | 90 days |
| Execution logs | Actions tab | 90 days (or per repo settings) |

## Reproducibility guarantees

| Aspect | Mechanism |
|--------|-----------|
| Tool versions | Pinned in Dockerfile build args (`JPEG_XL_VERSION`, `LIBAOM_VERSION`, etc.) |
| Encoding determinism | Single-threaded mode enforced for all encoders |
| Configuration | Study JSON files committed to the repository |
| Environment | Identical dev container image for local and CI runs |
| Traceability | Quality results JSON records tool versions, timestamps, and all parameters |

## Limitations

- **Runner variability**: GitHub-hosted runners may have different CPU
  microarchitectures between runs. This can affect encoding speed
  measurements but not quality metrics or compression ratios.
- **Dataset scope**: The workflow is configured to use `div2k-valid`
  (100 images). Larger datasets may exceed the 6-hour job timeout.
- **Storage**: The 14 GB runner disk limits the number of concurrent
  encoded artifacts. The pipeline discards encoded files after measurement
  by default.

## See also

- [Run studies on GitHub Actions](../how-to/run-studies-on-github-actions) — step-by-step guide for triggering studies
- [Architecture](architecture) — pipeline design and CI design decisions
- [Run the pipeline](../how-to/run-pipeline) — running studies locally
