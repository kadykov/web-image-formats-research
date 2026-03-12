---
title: "Generate visual comparisons"
description: "Create side-by-side comparison figures showing encoding artifacts at matched quality or file-size levels."
---

## Generate comparisons

After running the pipeline, generate visual comparison figures for a study:

```bash
just compare format-comparison
```

The comparison generator independently re-encodes the selected source image
(no encoded artifacts from the pipeline are needed) and assembles labeled
side-by-side grids together with Butteraugli distortion maps.

## What the comparison generator does

The generator works through the following steps:

1. **Image selection** — for each target group defined in the study config,
   selects the source image with the highest cross-format coefficient of
   variation (CV = std / mean) of the output metric (e.g. `bits_per_pixel`
   when targeting a quality score).  This maximises the *relative* spread
   of visible differences across encoding variants.
2. **Quality interpolation** — for every target value (e.g. SSIMULACRA2 = 70),
   interpolates the encoder quality setting per format using measurements from
   `quality.json`, then re-encodes the image to those settings on the fly.
3. **Fragment selection** — computes per-pixel Butteraugli distortion maps,
   builds an aggregate anisotropic standard-deviation map across all target
   values in the group, and uses it to pick the single most informative
   crop region for the whole group.
4. **Figure assembly** — assembles labeled comparison grids using ImageMagick
   `montage`, plus distortion-map grids and an annotated original for each group.

## Two types of comparison figures

For each target group the generator produces two kinds of figure:

| Figure type | Target metric | Purpose |
|-------------|---------------|---------|
| **Matched visual quality** | `ssimulacra2`, `psnr`, `ssim`, `butteraugli` | Show artifact character at equal perceived quality |
| **Matched file size** | `bits_per_pixel` | Show quality differences under equal file-weight constraints |

A single study run can yield both figure types simultaneously when the study
config lists target groups of both kinds.

## What it produces

Output goes to `data/analysis/<study-id>/comparison/`.

For each target group a set of figures is created:

- **`comparison_<metric>_<value>.png`** — crop grid at the target value
- **`distmap_<metric>_<value>.png`** — Butteraugli distortion-map grid at the target value
- **`original_annotated_<metric>.png`** — source image with the selected fragment highlighted

## Prerequisites

- `quality.json` must exist for the study: run `just pipeline <study-id> <budget>` first.
- The source dataset images must be present on disk: run `just fetch <dataset-id>` first.
- Comparison configuration (target values, tile parameter, excluded images) is
  read from the study JSON in `config/studies/`.

Unlike the main pipeline the comparison script is **fully self-contained**:
it re-encodes images independently and does not depend on any encoded
artifacts saved by the pipeline.

## Advanced options

```bash
# Use a larger crop region (default: 128 px before zoom)
python3 scripts/generate_comparison.py format-comparison --crop-size 192

# Change zoom factor (default: 3×)
python3 scripts/generate_comparison.py format-comparison --zoom 4

# Override the parameter that creates tiles within each figure
python3 scripts/generate_comparison.py format-comparison --tile-parameter format

# Pin to a specific source image instead of auto-selection
python3 scripts/generate_comparison.py format-comparison --source-image data/preprocessed/0801.png

# Custom output directory
python3 scripts/generate_comparison.py format-comparison --output data/analysis/custom-dir

# List studies that have quality measurements available
python3 scripts/generate_comparison.py --list
```

## Configuring comparison targets in the study file

Add a `comparison_targets` section to the study JSON to control which
figures are produced:

```json
{
  "comparison_targets": [
    { "metric": "ssimulacra2", "values": [60, 75, 90] },
    { "metric": "bits_per_pixel", "values": [0.5, 1.0, 1.5] }
  ],
  "comparison_tile_parameter": "format",
  "comparison_exclude_images": ["problematic_image.png"]
}
```

When no targets are configured, the generator defaults to
`ssimulacra2 = [60, 70, 80]` and `bits_per_pixel = [0.5, 1.0, 1.5]`.

## See also

- [Run the pipeline](run-pipeline) — encode and measure to produce `quality.json`
- [Generate reports](generate-report) — comparison images are embedded in the HTML report with a PhotoSwipe lightbox viewer
- [Architecture](../explanation/architecture) — comparison module design
