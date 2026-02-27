---
title: "Getting started"
description: "Set up the dev environment, run a format comparison study, and generate your first report."
---

This tutorial walks you through setting up the environment and running
a complete image format comparison — from dataset download to interactive report.

## Prerequisites

- [VS Code](https://code.visualstudio.com/) with the
  [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://www.docker.com/) installed and running

## Step 1: Set up the environment

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

4. **Run the quality checks**:

   ```bash
   just check
   ```

   This runs formatting checks, linting, type checking, and all tests.
   Everything should pass in a fresh dev container.

## Step 2: Fetch a dataset

Studies need source images. Fetch the DIV2K validation dataset (100 images, ~450 MB):

```bash
just fetch div2k-valid
```

For higher resolution research, you can also fetch 4K datasets
(see [Fetch Datasets](../how-to/fetch-datasets) for all options).

## Step 3: Run a study

Run the format comparison study, which encodes each image as JPEG, WebP, AVIF,
and JPEG XL and measures quality metrics. Give it a 30-minute time budget:

```bash
just pipeline format-comparison 30m
```

The pipeline will:

1. Pick images from the dataset one at a time
2. Encode each image in all configured formats and quality levels
3. Measure SSIMULACRA2, PSNR, SSIM, and Butteraugli for every encoded variant
4. Save results to `data/metrics/format-comparison/quality.json`
5. Repeat until the 30-minute budget runs out

## Step 4: Analyze results

Generate statistical summaries and static plots:

```bash
just analyze format-comparison
```

This creates CSV statistics and SVG plots in `data/analysis/format-comparison/`.

## Step 5: Generate visual comparisons

Generate side-by-side comparison images showing the worst-case encoding regions
with Butteraugli distortion maps:

```bash
just compare format-comparison
```

## Step 6: Generate an interactive report

Combine everything into an interactive HTML report with Plotly visualizations:

```bash
just report
```

Preview it locally:

```bash
just serve-report
```

Open <http://localhost:8000> in your browser to explore rate-distortion curves,
quality-vs-parameter plots, and comparison images.

## Next steps

- [Run the pipeline](../how-to/run-pipeline) — time budgets, advanced options
- [Fetch datasets](../how-to/fetch-datasets) — all supported datasets
- [Analyze results](../how-to/analyze-results) — understand the CSV and plots
- [Generate comparisons](../how-to/generate-comparison) — visual comparison options
- [Generate reports](../how-to/generate-report) — interactive HTML reports
- [Architecture](../explanation/architecture) — design decisions and rationale
