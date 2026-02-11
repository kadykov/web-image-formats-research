# How-to: Run the Full Pipeline

This guide shows how to run the complete image format comparison pipeline.

## Quick Start

Run a complete study (encoding + quality measurement):

```bash
# List available studies
just list-studies

# Run a complete pipeline for a study
just pipeline avif-quality-sweep
```

This will:

1. Encode images according to the study configuration
2. Measure quality metrics (SSIMULACRA2, PSNR, SSIM, Butteraugli)
3. Save results to `data/encoded/<study-id>/` and `data/metrics/<study-id>/`

## Step-by-Step

### 1. Fetch Dataset

First, ensure the required dataset is downloaded:

```bash
# List available datasets
just list-available-datasets

# Fetch a specific dataset
just fetch div2k-valid
```

### 2. Run Encoding Study

Encode images according to a study configuration:

```bash
# Preview what will be encoded
just dry-run-study avif-quality-sweep

# Run the encoding
just run-study avif-quality-sweep
```

This produces:

- Encoded images in `data/encoded/<study-id>/<format>/`
- Results JSON in `data/encoded/<study-id>/results.json`

### 3. Measure Quality

Measure quality metrics for all encoded images:

```bash
# Measure for a specific study
just measure-study avif-quality-sweep

# Or specify the results file
just measure data/encoded/avif-quality-sweep/results.json
```

This produces:

- Quality metrics JSON in `data/metrics/<study-id>/quality.json`

### 4. Analyze Results (Coming Soon)

```bash
just analyze
```

## Available Studies

List all configured studies:

```bash
just list-studies
```

Common studies:

- `avif-quality-sweep` — AVIF quality parameter sweep
- `format-comparison` — Compare JPEG, WebP, AVIF, JPEG XL
- `resolution-impact` — Test different resolutions

## Output Structure

After running a complete pipeline:

```text
data/
├── datasets/
│   └── DIV2K_valid/           # Downloaded datasets
├── preprocessed/
│   └── resolution-impact/      # Preprocessed (resized) images
├── encoded/
│   └── avif-quality-sweep/
│       ├── results.json        # Encoding results
│       └── avif/               # Encoded images
└── metrics/
    └── avif-quality-sweep/
        └── quality.json        # Quality measurements
```

## Customization

### Custom Worker Count

Speed up encoding/measurement with more workers:

```bash
# Run with 16 workers
python3 scripts/encode_images.py avif-quality-sweep
python3 scripts/measure_quality.py \
    data/encoded/avif-quality-sweep/results.json \
    --workers 16
```

### Custom Output Paths

```bash
# Custom encoding output
python3 scripts/encode_images.py avif-quality-sweep \
    --output custom/path/results.json

# Custom quality output
python3 scripts/measure_quality.py \
    data/encoded/avif-quality-sweep/results.json \
    --output custom/path/quality.json
```

## See Also

- [Fetch Datasets](fetch-datasets.md) — Download image datasets
- [Measure Quality](measure-quality.md) — Quality measurement details
- [Configuration Reference](../reference/configuration.md) — Study configuration
- [Data Structure](../reference/data-structure.md) — Output formats
