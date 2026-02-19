---
title: "Getting started"
description: "Quickstart to set up the dev environment, run the pipeline on a small dataset, and generate your first report."
---

This tutorial walks you through setting up the development environment
and running your first image format comparison.

## Prerequisites

- [VS Code](https://code.visualstudio.com/) with the
  [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://www.docker.com/) installed and running

## Setting Up the Environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kadykov/web-image-formats-research.git
   cd web-image-formats-research
   ```

2. **Open in VS Code and start the dev container**:

   ```bash
   code .
   ```

   VS Code will detect the `.devcontainer/` configuration and prompt you
   to "Reopen in Container". Click it. The first build takes several minutes
   because it compiles image encoding tools from source.

3. **Verify the setup**:

   ```bash
   just verify-tools
   ```

   You should see checkmarks for all encoding tools (cjpeg, cwebp, avifenc, cjxl)
   and quality measurement tools (ssimulacra2, butteraugli_main, ffmpeg).

4. **Run the test suite**:

   ```bash
   just check
   ```

   This runs linting, type checking, and all tests.

## Your First Encoding

Open a Python terminal and try encoding a test image:

```python
from pathlib import Path
from src.encoder import ImageEncoder

encoder = ImageEncoder(Path("results/encoded"))

# Encode a PNG to WebP at quality 85
result = encoder.encode_webp(Path("tests/fixtures/test_image.png"), quality=85)
print(f"Success: {result.success}")
print(f"File size: {result.file_size} bytes")
```

## Your First Quality Measurement

```python
from pathlib import Path
from src.quality import QualityMeasurer

measurer = QualityMeasurer()
metrics = measurer.measure_all(
    Path("tests/fixtures/test_image.png"),
    Path("results/encoded/test_image_q85.webp"),
)

print(f"PSNR: {metrics.psnr:.2f} dB")
print(f"SSIM: {metrics.ssim:.4f}")
if metrics.ssimulacra2 is not None:
    print(f"SSIMULACRA2: {metrics.ssimulacra2:.2f}")
if metrics.butteraugli is not None:
    print(f"Butteraugli: {metrics.butteraugli:.4f}")
```

## Fetching Test Datasets

Before running full comparisons, you'll need test images. First, see what's available:

```bash
just list-available-datasets
```

Then fetch the DIV2K validation dataset:

```bash
just fetch div2k-valid
```

This downloads 100 high-quality 2K images (~500MB). For more details, see the
[Fetch Datasets guide](../how-to/fetch-datasets).

## Running Your First Study

Now that you have a dataset, run a complete study with the unified pipeline:

```bash
# List available studies
just list-studies

# Run a format comparison for 30 minutes
just pipeline-analyze format-comparison 30m
```

The unified pipeline will:

1. Process images one-by-one until the 30-minute budget is exhausted
2. Encode each image in all configured formats (JPEG, WebP, AVIF, JPEG XL)
3. Measure quality metrics for each encoded variant
4. Save results to `data/metrics/format-comparison/quality.json`
5. Run analysis and generate plots in `data/analysis/format-comparison/`

You can view the results interactively:

```bash
# Generate HTML report
just report

# Serve locally (opens in browser)
just serve-report
```

## Next Steps

- Learn how to [run the unified pipeline](../how-to/run-pipeline) with time budgets
- See how to [fetch datasets](../how-to/fetch-datasets) for research
- Read the [How-to guides](../how-to/) for specific tasks
- See the [Reference](../reference/) for detailed module and tool documentation
- Check the [Explanation](../explanation/) section for design decisions
  and quality metric background
