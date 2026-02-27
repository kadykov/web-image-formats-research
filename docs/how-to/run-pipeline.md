---
title: "Run the pipeline"
description: "Run encoding studies with the unified pipeline, control time budgets, and manage study data."
---

## Run a study

The pipeline encodes images and measures quality metrics in a single pass.
Both a study ID and a time budget are required:

```bash
just pipeline format-comparison 30m
```

This processes images one at a time — encoding all configured format variants and
measuring quality metrics for each — until the time budget runs out.

Results are saved to `data/metrics/<study-id>/quality.json`.

## Time budget format

| Example | Meaning |
|---------|---------|
| `30m` | 30 minutes |
| `2h` | 2 hours |
| `1h30m` | 1 hour 30 minutes |
| `3600` | 3600 seconds |
| `90s` | 90 seconds |

## Available studies

These study configurations ship in `config/studies/`:

| Study ID | Description |
|----------|-------------|
| `format-comparison` | Compare JPEG, WebP, AVIF, JPEG XL |
| `avif-speed-sweep` | AVIF speed parameter sweep |
| `avif-chroma-subsampling` | AVIF chroma subsampling comparison |
| `jxl-effort-sweep` | JPEG XL effort level comparison |
| `webp-method-sweep` | WebP method parameter sweep |
| `resolution-impact` | Impact of image resolution on quality |

## Prerequisites

Ensure the study's dataset is downloaded first:

```bash
just fetch div2k-valid
```

## Advanced options via the script

The `just pipeline` command runs with `--save-worst-image` enabled by default.
For additional control, use the script directly:

```bash
# Dry run — preview what would run without executing
python3 scripts/run_pipeline.py format-comparison --time-budget 30m --dry-run

# Save encoded artifacts to disk (normally discarded after measurement)
python3 scripts/run_pipeline.py format-comparison --time-budget 1h --save-artifacts

# Disable worst-image saving
python3 scripts/run_pipeline.py format-comparison --time-budget 30m --no-save-worst-image

# Control parallelism
python3 scripts/run_pipeline.py format-comparison --time-budget 30m --workers 8
```

## Output structure

```text
data/metrics/<study-id>/
└── quality.json        # All quality measurements
```

With `--save-artifacts`, encoded images are also saved:

```text
data/encoded/<study-id>/
├── jpeg/               # Encoded JPEG files
├── webp/               # Encoded WebP files
├── avif/               # Encoded AVIF files
└── jxl/                # Encoded JXL files
```

## Clean study data

Remove all generated data for a specific study:

```bash
just clean-study format-comparison
```

Remove all study data (preserves datasets):

```bash
just clean-studies
```

## Typical workflow

```bash
just fetch div2k-valid                  # 1. Get images
just pipeline format-comparison 30m     # 2. Encode + measure
just analyze format-comparison          # 3. Generate plots
just compare format-comparison          # 4. Visual comparisons
just report                             # 5. Interactive report
just serve-report                       # 6. View in browser
```

## See also

- [Fetch datasets](fetch-datasets) — download source images
- [Analyze results](analyze-results) — generate statistics and plots
- [Generate comparisons](generate-comparison) — visual comparison images
- [Configuration reference](../reference/configuration) — study config schema
- [Architecture](../explanation/architecture) — pipeline design rationale
