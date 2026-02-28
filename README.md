# Web Image Formats Research

[![CI](https://github.com/kadykov/web-image-formats-research/workflows/CI/badge.svg)](https://github.com/kadykov/web-image-formats-research/actions)
[![codecov](https://codecov.io/gh/kadykov/web-image-formats-research/branch/main/graph/badge.svg)](https://codecov.io/gh/kadykov/web-image-formats-research)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

Research project for determining optimal modern image formats for the web,
with a focus on AVIF and comparative analysis against JPEG, WebP, and JPEG XL.

## Goals

- Identify optimal quality settings for AVIF encoding
- Determine when to use chroma subsampling based on quality targets
- Compare AVIF effectiveness against JPEG, WebP, and JPEG XL
- Analyze compression efficiency using perceptual quality metrics

## Key Findings

**SSIMULACRA2** — AVIF and JPEG XL deliver comparable perceptual quality at
significantly lower file sizes than WebP and JPEG:

![Format comparison — SSIMULACRA2 vs bytes per pixel](https://github.kadykov.com/web-image-formats-research/report/data/format-comparison/format-comparison_ssimulacra2_vs_bytes_per_pixel.svg)

**Butteraugli** — under this distortion model JPEG XL consistently edges ahead
of AVIF, highlighting how metric choice affects codec rankings:

![Format comparison — Butteraugli vs bytes per pixel](https://github.kadykov.com/web-image-formats-research/report/data/format-comparison/format-comparison_butteraugli_vs_bytes_per_pixel.svg)

See the [full interactive report](https://github.kadykov.com/web-image-formats-research/report/)
for all studies and metrics.

## Quick Start

1. Open this repository in VS Code
2. Reopen in the dev container (VS Code will prompt)
3. Verify everything works:

   ```bash
   just verify-tools
   just check  # Runs lint, typecheck, and tests
   ```

See the [Getting Started tutorial](docs/tutorials/getting-started) for detailed setup instructions.

## Project Structure

```text
├── src/                     # Source code modules
│   ├── study.py             # Study configuration loading
│   ├── dataset.py           # Dataset fetching and management
│   ├── preprocessing.py     # Image preprocessing (resize, convert)
│   ├── encoder.py           # Format encoding (JPEG, WebP, AVIF, JXL)
│   ├── quality.py           # Quality measurement (SSIMULACRA2, PSNR, SSIM, Butteraugli)
│   ├── pipeline.py          # Unified pipeline with time-budget control
│   ├── analysis.py          # Statistical analysis and static plotting
│   ├── interactive.py       # Interactive Plotly visualizations
│   ├── comparison.py        # Visual comparison image generation
│   └── report_images.py     # Responsive image optimization for reports
├── scripts/                 # CLI entry points for each workflow step
├── config/                  # Configuration files (datasets, studies)
│   └── studies/             # Per-study JSON configurations
├── data/                    # All research data (git-ignored)
│   ├── datasets/            # Raw image datasets
│   ├── preprocessed/        # Preprocessed images
│   ├── encoded/             # Encoded images (JPEG, WebP, AVIF, JXL)
│   ├── metrics/             # Quality measurements (JSON)
│   ├── analysis/            # Analysis outputs (CSV, SVG plots)
│   └── report/              # Generated HTML reports
├── docs/                    # Documentation (Diátaxis framework)
├── .devcontainer/           # Dev container configuration
├── .github/workflows/       # CI pipeline
├── pyproject.toml           # Project configuration and dependencies
└── justfile                 # Development task runner
```

## Development Commands

```bash
# Development
just install-dev   # Install all dependencies (dev + production)
just check         # Run all quality checks (lint + typecheck + test)
just test          # Run tests
just lint          # Check code style
just lint-fix      # Fix auto-fixable lint issues
just format        # Format code and markdown
just typecheck     # Run type checking
just verify-tools  # Verify encoding and measurement tools

# Study Workflow
just fetch <dataset-id>              # Fetch a dataset (e.g., div2k-valid, liu4k-v1-valid)
just pipeline <study-id> <time>      # Run unified encode+measure pipeline (e.g., 30m, 1h)
just analyze <study-id>              # Analyze results and generate plots
just compare <study-id>              # Generate visual comparison images
just report                          # Generate interactive HTML report for all studies
just serve-report [port]             # Serve report locally (default: http://localhost:8000)

# Release
just release-notes                   # Generate release notes from study results
just release-assets                  # Prepare release assets (zip + CSV files)

# Cleanup
just clean                           # Clean Python cache and build artifacts
just clean-study <study-id>          # Remove all data for a specific study
just clean-studies                   # Remove all study data (preserves datasets)

# Documentation
just docs-generate                   # Generate docs from source files and Python docstrings
just docs-install                    # Install documentation site dependencies
just docs-dev                        # Start documentation dev server (http://localhost:4321)
just docs-build                      # Build optimized documentation site
just docs-preview                    # Preview built documentation
```

## Documentation

This project follows the [Diátaxis](https://diataxis.fr/) documentation framework:

- [**Tutorials**](docs/tutorials/) — Step-by-step guides to get started
- [**How-to guides**](docs/how-to/) — Solutions for specific tasks
- [**Reference**](docs/reference/) — Technical details and API descriptions
- [**Explanation**](docs/explanation/) — Background concepts and design decisions

### Building Documentation

The documentation is generated using [Astro Starlight](https://starlight.astro.build/) for optimal performance:

```bash
# Generate and preview locally
just docs-dev

# Build for production
just docs-build
```

The documentation includes automatically generated API reference from Python docstrings.

## License

See [LICENSE](LICENSE).
