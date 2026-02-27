---
title: "Analyze study results"
description: "Generate statistics, plots, and CSV exports from completed pipeline runs."
---

## Analyze a study

After running a pipeline, generate statistical analysis and plots:

```bash
just analyze format-comparison
```

This loads `data/metrics/format-comparison/quality.json`, computes statistics,
and writes outputs to `data/analysis/format-comparison/`.

## What it produces

- **`<study-id>_statistics.csv`** — aggregated statistics: mean, median,
  5th/25th/50th/75th/95th percentiles for all metrics, grouped by format
  and sweep parameters
- **Quality plots (SVG)** — SSIMULACRA2, PSNR, SSIM, Butteraugli vs the
  sweep parameter, showing mean and 5th-percentile (worst case)
- **Rate-distortion plots (SVG)** — quality metrics vs bytes per pixel,
  grouped by format
- **Efficiency plots (SVG)** — bytes per quality unit per pixel
- **Bytes-per-pixel plots (SVG)** — file size distribution with percentile bands

All plots are SVG vector graphics, suitable for papers and presentations.

## List studies with available results

```bash
python3 scripts/analyze_study.py --list
```

## Advanced options

Use the script directly for more control:

```bash
# Analyze by study ID
python3 scripts/analyze_study.py format-comparison

# Analyze from a specific quality.json path
python3 scripts/analyze_study.py data/metrics/format-comparison/quality.json

# Custom output directory
python3 scripts/analyze_study.py format-comparison --output data/analysis/custom
```

## Troubleshooting

- **No quality measurements found**: Run the pipeline first —
  `just pipeline <study-id> <time-budget>`
- **Missing plots**: Some plots require at least two distinct parameter values.
  Check the CSV statistics to see which metrics have data.

## See also

- [Run the pipeline](run-pipeline) — encode and measure before analyzing
- [Generate comparisons](generate-comparison) — worst-case visual comparisons
- [Generate reports](generate-report) — interactive HTML report
- [Tools reference](../reference/tools) — metric interpretation tables
- [Data structure reference](../reference/data-structure) — output file layout
