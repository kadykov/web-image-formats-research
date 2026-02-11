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

## Quick Start

1. Open this repository in VS Code
2. Reopen in the dev container (VS Code will prompt)
3. Verify everything works:

   ```bash
   just verify-tools
   just check  # Runs lint, typecheck, and tests
   ```

See the [Getting Started tutorial](docs/tutorials/getting-started.md) for detailed setup instructions.

## Project Structure

```text
├── src/                     # Source code modules
│   ├── dataset.py           # Dataset fetching and management
│   ├── preprocessing.py     # Image preprocessing (resize, convert)
│   ├── encoder.py           # Format encoding (JPEG, WebP, AVIF, JXL)
│   ├── quality.py           # Quality measurement (SSIMULACRA2, PSNR, SSIM, Butteraugli)
│   └── analysis.py          # Analysis and visualization
├── tests/                   # Unit and integration tests
├── scripts/                 # Executable pipeline scripts
├── config/                  # Configuration files (datasets, encoding, etc.)
├── data/                    # All research data
│   ├── datasets/            # Raw image datasets
│   ├── preprocessed/        # Preprocessed images
│   ├── encoded/             # Encoded images (JPEG, WebP, AVIF, JXL)
│   ├── metrics/             # Quality measurements (JSON/CSV)
│   └── analysis/            # Analysis outputs (plots, reports)
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
just format        # Format code
just typecheck     # Run type checking
just verify-tools  # Verify encoding and measurement tools

# Dataset Management
just fetch <dataset-id>          # Fetch a dataset (e.g., div2k-valid, div2k-train)
just list-available-datasets     # List all datasets in configuration
just list-datasets               # List downloaded datasets

# Encoding Studies
just list-studies                # List available study configurations
just run-study <study-id>        # Run an encoding study
just run-study-clean <study-id>  # Clean existing data then run study
just dry-run-study <study-id>    # Preview what a study would do

# Quality Measurement
just measure <results-file>      # Measure quality for encoded images
just measure-study <study-id>    # Measure quality for a specific study
just measure-with-workers <results-file> <workers>  # Use custom worker count

# Analysis
just analyze <study-id>          # Analyze study results and generate plots
just analyze-to <study-id> <dir> # Analyze with custom output directory
just list-analyzed               # List studies available for analysis

# Complete Pipeline
just pipeline <study-id>         # Run encode + measure + analyze for a study
just pipeline-clean <study-id>   # Run complete pipeline starting fresh

# Cleanup
just clean                       # Clean Python cache and build artifacts
just clean-study <study-id>      # Remove all data for a specific study
just clean-studies               # Remove all study data (preserves datasets)
just clean-all-data              # Remove ALL data including datasets (careful!)
```

## Documentation

This project follows the [Diátaxis](https://diataxis.fr/) documentation framework:

- [**Tutorials**](docs/tutorials/) — Step-by-step guides to get started
- [**How-to guides**](docs/how-to/) — Solutions for specific tasks
- [**Reference**](docs/reference/) — Technical details and API descriptions
- [**Explanation**](docs/explanation/) — Background concepts and design decisions

## License

See [LICENSE](LICENSE).
