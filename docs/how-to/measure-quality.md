---
title: "Measure image quality"
description: "How quality metrics are measured and where to find the results."
---

Quality measurement is **integrated into the pipeline** — there is no separate
measurement step. When you run `just pipeline`, each encoded image is measured
automatically before moving to the next source image.

## How it works

For every encoded variant, the pipeline measures four metrics:

- **SSIMULACRA2** — perceptual quality (higher is better)
- **PSNR** — peak signal-to-noise ratio in dB (higher is better)
- **SSIM** — structural similarity, 0–1 (higher is better)
- **Butteraugli** — perceptual distance (lower is better)

Results are saved to `data/metrics/<study-id>/quality.json`.

## Running measurements

```bash
just pipeline format-comparison 30m
```

This encodes and measures in a single pass. See
[Run the pipeline](run-pipeline) for time budgets and advanced options.

## Verifying tools

The measurement tools must be available in the environment:

```bash
just verify-tools
```

All tools (ssimulacra2, butteraugli_main, ffmpeg) are pre-installed in
the dev container.

## See also

- [Run the pipeline](run-pipeline) — the main command for encoding + measurement
- [Analyze results](analyze-results) — generate statistics and plots from measurements
- [Tools reference](../reference/tools) — metric interpretation tables and CLI usage
