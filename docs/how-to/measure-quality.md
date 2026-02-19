---
title: "Measure image quality"
description: "How to run objective quality metrics (PSNR, SSIM, SSIMULACRA2, Butteraugli) and interpret the output."
---

This guide shows how to measure perceptual quality metrics for encoded images.

## Prerequisites

- Encoded images from a study run (see [Run Pipeline](run-pipeline))
- Quality measurement tools installed (`ssimulacra2`, `butteraugli_main`, `ffmpeg`)

Verify tools are available:

```bash
just verify-tools
```

## Basic Usage

After running an encoding study, measure quality metrics:

```bash
# Measure quality for a specific study
just measure-study avif-quality-sweep

# Or specify the results file directly
just measure data/encoded/avif-quality-sweep/results.json
```

This will:

1. Load the encoding results JSON
2. Measure SSIMULACRA2, PSNR, SSIM, and Butteraugli for each encoded image
3. Save results to `data/metrics/<study-id>/quality.json`

## Custom Worker Count

Control parallelism for faster measurements:

```bash
# Use 16 parallel workers
just measure-with-workers data/encoded/format-comparison/results.json 16
```

Default is your CPU count.

## Output Location

By default, results are saved to:

```text
data/metrics/<study-id>/quality.json
```

Override with `--output`:

```bash
python3 scripts/measure_quality.py \
    data/encoded/avif-quality-sweep/results.json \
    --output custom/path/metrics.json
```

## Understanding the Output

The output JSON contains quality measurements for each encoded image:

```json
{
  "study_id": "avif-quality-sweep",
  "study_name": "AVIF Quality Sweep",
  "dataset": { ... },
  "timestamp": "2026-02-11T10:30:00.000000+00:00",
  "measurements": [
    {
      "encoded_path": "data/encoded/avif-quality-sweep/avif/image_q75.avif",
      "original_image": "data/datasets/DIV2K_valid/0801.png",
      "format": "avif",
      "quality": 75,
      "file_size": 123456,
      "ssimulacra2": 78.5,
      "psnr": 42.3,
      "ssim": 0.978,
      "butteraugli": 0.85,
      ...
    }
  ]
}
```

### Metric Interpretation

- **SSIMULACRA2**: Higher is better
  - `>90` — Excellent (near-lossless)
  - `70-90` — Good (typical web quality)
  - `<30` — Poor
  
- **PSNR**: Higher is better (in dB)
  - `>40 dB` — Excellent
  - `30-40 dB` — Good
  - `<30 dB` — Poor

- **SSIM**: Higher is better (0-1 scale)
  - `>0.95` — Excellent
  - `0.85-0.95` — Good
  - `<0.85` — Poor

- **Butteraugli**: Lower is better
  - `<1.0` — Excellent
  - `1.0-1.5` — Good
  - `>3.0` — Poor

## Complete Pipeline

Run encoding and measurement with the unified pipeline:

```bash
# Run for 1 hour with time budget
just pipeline-analyze avif-quality-sweep 1h
```

Or use the legacy separate stages:

```bash
# Encode then measure separately
just run-study avif-quality-sweep
just measure-study avif-quality-sweep
```

## Troubleshooting

### Measurement Tools Not Found

If you see errors like `ssimulacra2: command not found`:

```bash
# Verify tool availability
just verify-tools

# Install missing tools (in dev container, these should be pre-installed)
```

### Files Not Found

Ensure the encoding stage completed successfully:

```bash
# Check that results.json exists
ls -lh data/encoded/avif-quality-sweep/results.json

# Check that encoded images exist
ls data/encoded/avif-quality-sweep/avif/
```

### Measurement Errors

The script continues on errors and records them in the output:

```json
{
  "encoded_path": "...",
  "ssimulacra2": null,
  "measurement_error": "Original image not found: ..."
}
```

Check the summary at the end for failed measurements.

## Next Steps

- [Analyze Results](../explanation/architecture) — Visualize and compare metrics
- [Data Structure](../reference/data-structure) — Understand the output format
- [Configuration Reference](../reference/configuration) — Study and metric configuration
