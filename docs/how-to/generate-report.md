# How to Generate Interactive HTML Reports

This guide explains how to generate interactive HTML reports from quality measurement results.

## Overview

The report generator creates a static website with:

- **Interactive Plotly visualizations** — Zoom, pan, hover tooltips showing encoding parameters
- **Rate-distortion curves** — Most informative plots shown first
- **Data downloads** — CSV statistics and SVG figures for offline analysis
- **Tool version tracking** — Records versions of all encoding and measurement tools
- **Dark mode** — Responsive design that works on all devices

Reports are fully static (HTML + JavaScript) with no backend required, making them perfect for GitHub Pages or simple file sharing.

## Prerequisites

Before generating reports, you need quality measurement results:

```bash
# Run unified pipeline with time budget (includes analysis)
just pipeline-analyze format-comparison 30m

# Or run separate stages
just run-study format-comparison
just measure-study format-comparison
just analyze format-comparison
```

## Generating Reports

### Generate for All Studies

Generate reports for all studies with available quality measurements:

```bash
just report
```

Output: `data/report/index.html` and individual study pages

### Generate for Specific Studies

Generate reports for selected studies only:

```bash
just report-studies format-comparison avif-quality-sweep
```

### List Available Studies

See which studies have quality measurements ready for reporting:

```bash
just list-report-studies
```

### Serve Locally

Preview the generated report in your browser:

```bash
just serve-report       # Serves on http://localhost:8000
just serve-report 9000  # Custom port
```

Then open the URL in your browser.

## Report Structure

Generated reports are organized as follows:

```text
data/report/
├── index.html                          # Landing page with study list
├── format-comparison.html              # Individual study page
├── avif-quality-sweep.html
├── data/                               # Data downloads
│   ├── format-comparison/
│   │   ├── format-comparison_statistics.csv
│   │   ├── format-comparison_ssimulacra2_vs_bytes_per_pixel.svg
│   │   └── ... (more SVG figures)
│   └── avif-quality-sweep/
│       └── ...
└── assets/
    └── plotly-basic.min.js             # Plotly library (auto-downloaded)
```

## Understanding the Visualizations

### Rate-Distortion Plots (Most Important)

These appear **first** and show quality vs file size (bytes per pixel):

- **X-axis**: bytes_per_pixel (storage cost)
- **Y-axis**: quality metric (SSIMULACRA2, PSNR, etc.)
- **Higher and lefter is better**: High quality at low file size
- **Hover tooltips** show: format, quality setting, exact metrics, file size

Metrics are ordered by importance:

1. **SSIMULACRA2** — Best perceptual metric (correlates with human perception)
2. **Butteraugli** — Google's perceptual metric
3. **PSNR** — Traditional metric (less accurate for perception)
4. **SSIM** — Structural similarity (better than PSNR)

### Quality vs Parameter Plots

Show how quality metrics change with encoding parameters (quality setting, speed, effort):

- **Mean line** — Average across all images
- **Shaded region** — 5th percentile (worst-case images)
- Helps identify parameter sweet spots

### Efficiency Plots

Show compression efficiency (bytes per quality unit per pixel):

- **Lower is better** — Less storage needed per quality point
- Useful for comparing formats at similar quality levels

### Bytes Per Pixel Plots

Show file size distribution across quality levels:

- **Median line** — Typical file size
- **Percentile bands** — p05 (best compression) to p95 (worst compression)

## Interactive Features

### Zoom and Pan

- **Box zoom**: Click and drag to zoom into a region
- **Pan**: Hold shift and drag to pan
- **Reset**: Double-click to reset view

### Legend Toggles

- **Click legend items** to show/hide specific formats/parameters
- **Double-click** to isolate one item (hide all others)

### Hover Tooltips

Hover over data points to see:

- Format and quality setting
- Exact metric values
- File size in bytes and bytes/pixel
- Secondary parameters (speed, effort, chroma subsampling)

## Data Downloads

Each study page provides downloadable data:

### CSV Statistics

Complete statistics table with per-quality-level aggregates:

- Format, quality setting, image count
- Mean/median/p05/p95 for all metrics
- Bytes per pixel statistics
- Compression ratios

**Use case**: Custom analysis in Excel, R, Python, etc.

### SVG Figures

Static vector graphics versions of all plots:

- Perfect for papers and presentations
- Scalable to any size without quality loss
- Smaller file size than raster formats for plots

**Use case**: Embed in LaTeX documents, PowerPoint, or offline reports

## Tool Version Information

Each study page includes a collapsible "Tool Versions" section showing:

- **Encoders**: cjpeg, cwebp, avifenc, cjxl
- **Measurement tools**: ssimulacra2, butteraugli, ffmpeg

This ensures reproducibility — you know exactly which tool versions produced the results.

## Advanced Usage

### Custom Output Directory

```bash
python3 scripts/generate_report.py --output custom/path format-comparison
```

### Programmatic Access

```python
from scripts.generate_report import generate_report, discover_studies

# Generate for specific studies
studies = ["format-comparison", "avif-quality-sweep"]
generate_report(studies, output_dir="data/report")

# Auto-discover and generate for all
all_studies = discover_studies()
generate_report(all_studies, output_dir="data/report")
```

## Troubleshooting

### "No quality.json found for study"

The study hasn't completed quality measurement yet:

```bash
just measure-study <study-id>
```

### "No static figures found"

The analysis stage hasn't run yet:

```bash
just analyze <study-id>
```

### Plotly Figures Not Interactive

The plotly bundle might not have downloaded:

```bash
# Regenerate report (will auto-download)
just report
```

### Old Data Shown in Report

Regenerate reports after re-running measurements:

```bash
just report
```

## Publishing Reports

Reports are fully static and can be hosted anywhere:

### GitHub Pages

1. Commit `data/report/` to a `gh-pages` branch
2. Enable GitHub Pages in repository settings
3. Access at `https://username.github.io/repo-name/`

### Simple Web Server

Any static hosting works (Netlify, Vercel, S3, etc.):

```bash
# Upload the entire data/report/ directory
rsync -avz data/report/ user@server:/var/www/html/
```

### Local Preview

```bash
just serve-report 8080
# Open http://localhost:8080 in browser
```

## Related Documentation

- [How to Analyze Results](analyze-results.md) — Generate static matplotlib figures
- [How to Measure Quality](measure-quality.md) — Run quality measurements
- [Architecture](../explanation/architecture.md) — Overall system design
