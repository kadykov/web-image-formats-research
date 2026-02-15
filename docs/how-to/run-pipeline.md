# How-to: Run the Unified Pipeline

This guide shows how to run image encoding studies using the unified pipeline with time-budget control.

## Quick Start

Run a study with a time budget:

```bash
# Run for 30 minutes, then analyze  
just pipeline-analyze format-comparison 30m
```

This runs the unified pipeline (encode + measure for each image) for 30 minutes,
then automatically analyzes the collected data.

## Time Budget Format

Time budgets can be specified in several formats:

- `30m` — 30 minutes
- `2h` — 2 hours  
- `1h30m` — 1 hour and 30 minutes
- `3600` — 3600 seconds (1 hour)
- `90s` — 90 seconds

## Unified Pipeline Architecture

The unified pipeline processes each image atomically:

1. **Select next image** from the dataset
2. **Encode** all format variants for that image (in parallel)
3. **Measure** quality of all variants (in parallel)
4. **Save results** to quality JSON
5. **Repeat** until time budget is exhausted

This approach provides:

- **Predictable runtime**: Set time budget instead of guessing `max_images`
- **Progress guarantees**: Always completes full encode+measure for each image
- **Reduced disk I/O**: Uses `/dev/shm` for temporary storage
- **Error isolation**: Failures on one image don't block others

## Prerequisites

First, ensure the required dataset is downloaded:

```bash
# List available datasets
just list-available-datasets

# Fetch a specific dataset
just fetch div2k-valid
```

## Pipeline Commands

### Run Full Pipeline

```bash
# Encode + measure with time budget, then analyze
just pipeline-analyze format-comparison 30m

# Run all studies with their configured time budgets
just pipeline-all
```

### Run with Artifact Saving

By default, encoded images are stored in memory (`/dev/shm`) and discarded
after measurement. To save artifacts to disk:

```bash
# Save encoded images to data/encoded/<study-id>/
just pipeline-save avif-quality-sweep 1h

# Or use the CLI directly
python3 scripts/run_pipeline.py avif-quality-sweep 1h --save-artifacts
```

### Preview Without Running

```bash
# See what would be executed without running
just pipeline-dry-run format-comparison 30m

# Example output:
# Study: Format Comparison (format-comparison)
# Dataset: DIV2K Validation (div2k-valid) - 10 images max
# Time Budget: 1800.0 seconds (30 minutes)
# 
# Tasks per image: 16
#   - jpeg (quality: 60, 75, 85, 95)
#   - webp (quality: 60, 75, 85, 95)
#   - avif (quality: 60, 75, 85, 95; speed: 4)
#   - jxl (quality: 60, 75, 85, 95)
```

### Control Worker Count

```bash
# Use more workers for parallelism
python3 scripts/run_pipeline.py format-comparison 30m --workers 16
```

### Clean Pipeline Data

```bash
# Remove metrics for a study
just pipeline-clean format-comparison
```

## Output Structure

After running the pipeline, your data directory will look like:

```text
data/
└── metrics/
    └── format-comparison/
        └── quality.json        # Quality measurements (encoding + metrics)
```

If using `--save-artifacts`, you'll also see:

```text
data/
├── encoded/
│   └── format-comparison/
│       ├── jpeg/           # Encoded JPEG files
│       ├── webp/           # Encoded WebP files
│       ├── avif/           # Encoded AVIF files
│       └── jxl/            # Encoded JXL files
└── metrics/
    └── format-comparison/
        └── quality.json
```

## Study Configuration

Studies can optionally specify a default `time_budget` in their configuration:

```json
{
  "id": "format-comparison",
  "name": "Format Comparison",
  "time_budget": 1800,
  "dataset": { "id": "div2k-valid", "max_images": 10 },
  "encoders": [
    { "format": "jpeg", "quality": [60, 75, 85, 95] }
  ]
}
```

Time budget can be overridden on the command line:

```bash
# Use configured time_budget (1800 seconds)
just pipeline format-comparison

# Override with 1 hour
just pipeline format-comparison 1h
```

Common configured studies:

- `avif-quality-sweep` — AVIF quality parameter sweep
- `format-comparison` — Compare JPEG, WebP, AVIF, JPEG XL
- `jxl-effort-sweep` — JPEG XL effort level comparison
- `resolution-impact` — Test different image resolutions

## Using the Legacy Separate Pipeline

For backward compatibility, the old separate encode → measure workflow is still available:

```bash
# Run separate stages
just pipeline-separate format-comparison

# Or manually:
just run-study format-comparison
just measure-study format-comparison
just analyze format-comparison
```

The separate pipeline is useful when you need to:

- Inspect encoded files before measurement
- Run measurement with different settings
- Debug encoding issues independently

## Related Documentation

- [Fetch Datasets](fetch-datasets.md) — Download image datasets
- [Configuration Reference](../reference/configuration.md) — Study configuration with time budgets
- [Architecture](../explanation/architecture.md) — Pipeline design rationale
- [Data Structure](../reference/data-structure.md) — Output formats
