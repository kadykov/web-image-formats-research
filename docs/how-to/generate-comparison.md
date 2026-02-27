---
title: "Generate visual comparisons"
description: "Create side-by-side comparison images showing worst-case encoding regions with distortion maps."
---

## Generate comparisons

After running the pipeline, generate visual comparison images for a study:

```bash
just compare format-comparison
```

This finds the worst-case encoded images, crops the most distorted regions,
and creates comparison montages with Butteraugli distortion maps.

## Prerequisites

The pipeline must have been run with `--save-worst-image` (the default when
using `just pipeline`). This saves the source images needed for comparison.

## What it produces

For each study, the comparison generator:

1. Identifies the worst-performing source image (by average or variance of the chosen metric)
2. Finds the most distorted region using a sliding-window search
3. Generates per-format Butteraugli distortion maps
4. Assembles a side-by-side montage: original crop, encoded crops, distortion maps

Output goes to `data/analysis/<study-id>/comparisons/`.

## Advanced options

Use the script directly for more control:

```bash
# Custom crop size (default: 512)
python3 scripts/generate_comparison.py format-comparison --crop-size 256

# Custom zoom factor (default: 2)
python3 scripts/generate_comparison.py format-comparison --zoom 3

# Use a different metric for worst-image selection
python3 scripts/generate_comparison.py format-comparison --metric ssimulacra2

# Selection strategy: average (default), variance, or both
python3 scripts/generate_comparison.py format-comparison --strategy both

# Compare a specific source image instead of auto-selecting worst
python3 scripts/generate_comparison.py format-comparison --source-image 0801.png
```

## See also

- [Run the pipeline](run-pipeline) — encode and measure (generates worst-image data)
- [Generate reports](generate-report) — comparison images are included in the HTML report
- [Architecture](../explanation/architecture) — comparison module design
