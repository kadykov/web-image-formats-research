# How to Analyze Study Results

This guide explains how to analyze quality measurement results and generate plots for a study.

## Prerequisites

- Completed quality measurements for a study (see [How to Measure Quality](measure-quality.md))
- The study must have a `quality.json` file in `data/metrics/<study-id>/`

## Basic Usage

### Analyze a Study

To analyze a study and generate plots:

```bash
just analyze <study-id>
```

Example:

```bash
just analyze avif-quality-sweep
```

This will:

1. Load quality measurements from `data/metrics/<study-id>/quality.json`
2. Compute statistical aggregates (mean, 5th/25th/50th/75th/95th percentiles)
3. Generate CSV with all statistics
4. Create visualization plots in SVG format (vector graphics)

### Outputs

Analysis outputs are saved to `data/analysis/<study-id>/`:

- **`<study-id>_statistics.csv`** - Complete statistical summary with all metrics
- **Quality metric plots** - SSIMULACRA2, PSNR, SSIM, Butteraugli vs the sweep parameter
- **Rate-distortion plots** - Quality metrics vs bytes per pixel
- **Efficiency plots** - Bytes per quality score per pixel
- **Bytes per pixel plots** - File size distribution with percentile bands

All plots are saved as **SVG** (Scalable Vector Graphics) for:

- Perfect quality at any zoom level
- Smaller file size for plots with few data points
- Easy embedding in papers and presentations

Each plot shows:

- **Mean values** - Solid lines with filled markers
- **5% worst values** - Dashed lines with open markers (quality plots only)

## Available Commands

### List Available Studies

See which studies have quality measurements ready for analysis:

```bash
just list-analyzed
```

### Custom Output Directory

Specify a custom output location:

```bash
just analyze-to <study-id> <output-directory>
```

Example:

```bash
just analyze-to avif-quality-sweep data/analysis/custom-output
```

### Run Complete Pipeline

Run encoding, quality measurement, and analysis in one command:

```bash
just pipeline <study-id>
```

Or start fresh by cleaning existing data first:

```bash
just pipeline-clean <study-id>
```

## Understanding the Outputs

### Statistics CSV

The CSV contains aggregated statistics for each combination of parameters swept in the study.

Key columns:

- **Grouping columns** - `format`, `quality`, `chroma_subsampling`, `speed`, `resolution`
- **Quality metrics** - `ssimulacra2_mean`, `psnr_mean`, `ssim_mean`, `butteraugli_mean`
- **Percentiles** - `*_p05` (5% worst), `*_p25`, `*_p50` (median), `*_p75`, `*_p95`
- **Efficiency metrics** - `bytes_per_pixel_mean`, `bytes_per_ssimulacra2_per_pixel_mean`

### Quality Plots

Quality plots show perceptual similarity metrics:

- **SSIMULACRA2** - Perceptual quality (higher is better, >90 = excellent)
- **PSNR** - Peak Signal-to-Noise Ratio (higher is better, >40dB = excellent)
- **SSIM** - Structural Similarity (0-1, higher is better, >0.95 = excellent)
- **Butteraugli** - Perceptual distance (lower is better, <1.0 = excellent)

### Efficiency Plots

Efficiency plots help compare encoders and settings:

- **bytes_per_pixel** - File size normalized by resolution
- **bytes_per_ssimulacra2_per_pixel** - How many bytes needed per pixel to achieve quality
  - Lower is better = more efficient encoding
- **bytes_per_psnr_per_pixel** - Similar but using PSNR metric
- **bytes_per_ssim_per_pixel** - Similar but using SSIM metric

## Examples

### Analyze AVIF Quality Sweep

```bash
# Run analysis
just analyze avif-quality-sweep

# View CSV statistics
head -20 data/analysis/avif-quality-sweep/avif-quality-sweep_statistics.csv

# List all generated plots
ls data/analysis/avif-quality-sweep/*.svg
```

### Compare Multiple Formats

```bash
# Analyze format comparison study
just analyze format-comparison

# The plots will show different formats side by side
# allowing direct visual comparison of:
# - Quality at each setting
# - Encoding efficiency
```

### Analyze Resolution Impact

```bash
# Analyze resolution impact study
just analyze resolution-impact

# Plots will use resolution as x-axis
# showing how quality/efficiency varies with image size
```

## Advanced Usage

### Direct Script Usage

You can also use the analysis script directly:

```bash
# Analyze by study ID
python3 scripts/analyze_study.py avif-quality-sweep

# Analyze with explicit quality.json path
python3 scripts/analyze_study.py data/metrics/avif-quality-sweep/quality.json

# Custom output directory
python3 scripts/analyze_study.py avif-quality-sweep --output data/analysis/my-analysis

# List available studies
python3 scripts/analyze_study.py --list

# Get help
python3 scripts/analyze_study.py --help
```

## Troubleshooting

### No Quality Measurements Found

If you get an error about missing quality measurements:

```bash
# First run quality measurements
just measure-study <study-id>

# Then run analysis
just analyze <study-id>
```

### Missing Plots

If some plots are missing, it may be because:

- The metric wasn't measured (e.g., Butteraugli may be unavailable)
- The parameter doesn't vary in your study (need at least 2 different values)

Check the CSV statistics file - any metric with data will have a corresponding plot.

## See Also

- [How to Run the Pipeline](run-pipeline.md) - Complete workflow from encoding to analysis
- [How to Measure Quality](measure-quality.md) - Quality measurement step
- [Configuration Reference](../reference/configuration.md) - Study configuration options
- [Data Structure Reference](../reference/data-structure.md) - Understanding output formats
