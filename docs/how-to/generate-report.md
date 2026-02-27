---
title: "Generate reports"
description: "Build interactive HTML reports from study results and serve them locally."
---

## Generate a report

After running the pipeline and analysis, generate an interactive HTML report
covering all studies with available data:

```bash
just report
```

Output: `data/report/index.html` and individual study pages.

## Serve locally

Preview the report in your browser:

```bash
just serve-report
```

Opens at <http://localhost:8000>. To use a custom port:

```bash
just serve-report 9000
```

## Prerequisites

The report generator needs:

1. Quality measurements (`data/metrics/<study-id>/quality.json`) — from [running the pipeline](run-pipeline)
2. Analysis outputs (`data/analysis/<study-id>/`) — from [analyzing results](analyze-results)
3. Comparison images (optional) — from [generating comparisons](generate-comparison)

## Report contents

Each study page includes:

- **Interactive Plotly visualizations** — rate-distortion curves, quality-vs-parameter plots,
  efficiency plots with zoom, pan, and hover tooltips
- **Static SVG figures** — downloadable vector plots from the analysis step
- **CSV statistics** — downloadable per-study statistics tables
- **Comparison images** — worst-case visual comparisons (if generated)
- **Tool versions** — encoder and measurement tool versions for reproducibility

## Output structure

```text
data/report/
├── index.html                    # Landing page with study list
├── format-comparison.html        # Individual study page
├── ...
├── data/                         # Downloadable data per study
│   └── <study-id>/
│       ├── *_statistics.csv
│       └── *.svg
└── assets/
    └── plotly-basic.min.js       # Plotly library (auto-downloaded)
```

## Troubleshooting

- **"No quality.json found"**: Run the pipeline first — `just pipeline <study-id> <time>`
- **"No static figures found"**: Run analysis — `just analyze <study-id>`
- **Plotly not interactive**: Regenerate the report — `just report` re-downloads the Plotly bundle

## See also

- [Run the pipeline](run-pipeline) — encode and measure
- [Analyze results](analyze-results) — generate statistics and static plots
- [Generate comparisons](generate-comparison) — visual comparison images
- [Architecture](../explanation/architecture) — report generation design
