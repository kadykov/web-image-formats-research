---
title: "Create a custom study"
description: "Define a new encoding experiment by creating a study configuration file — no code changes required."
---

A study is a JSON file in `config/studies/` that defines which formats,
quality levels, and encoder parameters to test. Creating a new study
requires no code changes.

## Create a study file

Create a new file in `config/studies/`. The filename (without `.json`)
becomes the study ID used in all CLI commands:

```bash
# config/studies/my-study.json → study ID is "my-study"
```

### Minimal example

Compare AVIF and WebP at a few quality levels:

```json
{
  "$schema": "../study.schema.json",
  "id": "my-study",
  "name": "My Custom Study",
  "description": "Compare AVIF and WebP at representative quality levels.",
  "dataset": {
    "id": "div2k-valid",
    "max_images": 10
  },
  "encoders": [
    {
      "format": "avif",
      "quality": [50, 65, 80],
      "speed": 4
    },
    {
      "format": "webp",
      "quality": [50, 65, 80],
      "method": 4
    }
  ]
}
```

### Parameter sweep example

Sweep quality as a range and test multiple speed settings:

```json
{
  "$schema": "../study.schema.json",
  "id": "avif-deep-sweep",
  "name": "AVIF Deep Quality Sweep",
  "description": "Fine-grained quality sweep with speed variants.",
  "dataset": {
    "id": "div2k-valid",
    "max_images": 20
  },
  "encoders": [
    {
      "format": "avif",
      "quality": { "start": 30, "stop": 90, "step": 5 },
      "speed": [2, 4, 6]
    }
  ]
}
```

The range `{"start": 30, "stop": 90, "step": 5}` expands to
`[30, 35, 40, 45, ..., 85, 90]`.

### Multi-resolution example

Test how resolution affects encoding efficiency:

```json
{
  "$schema": "../study.schema.json",
  "id": "resolution-test",
  "name": "Resolution Impact Test",
  "description": "Compare encoding at different target resolutions.",
  "dataset": {
    "id": "div2k-valid",
    "max_images": 10
  },
  "encoders": [
    {
      "format": "avif",
      "quality": [50, 65, 80],
      "speed": 4,
      "resolution": [1920, 1280, 640]
    }
  ]
}
```

## Study configuration fields

### Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Must match the filename (without `.json`). Pattern: `^[a-z0-9][a-z0-9_-]*$` |
| `name` | string | Yes | Human-readable name. |
| `description` | string | No | Purpose of the study. |
| `time_budget` | number | No | Default time budget in seconds. Overridden by CLI `--time-budget`. |
| `dataset.id` | string | Yes | Dataset identifier from `config/datasets.json`. |
| `dataset.max_images` | integer | No | Limit images used from the dataset. |
| `encoders` | array | Yes | List of encoder configurations (at least one). |

### Encoder fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format` | string | Yes | `"jpeg"`, `"webp"`, `"avif"`, or `"jxl"`. |
| `quality` | int/list/range | Yes | Quality settings (0–100). See formats below. |
| `speed` | int/int[] | No | AVIF speed (0–10). 0 = slowest/best, 10 = fastest. |
| `effort` | int/int[] | No | JXL effort (1–10). Higher = slower/better. |
| `method` | int/int[] | No | WebP method (0–6). Higher = slower/better. |
| `chroma_subsampling` | string[] | No | Modes: `"444"`, `"422"`, `"420"`, `"400"`. |
| `resolution` | int/int[] | No | Target resolution(s) in pixels (longest edge). |
| `extra_args` | object | No | Additional CLI arguments as key-value pairs. |

### Quality specification formats

| Format | Example | Expands to |
|--------|---------|------------|
| Single integer | `75` | `[75]` |
| Explicit list | `[60, 75, 90]` | `[60, 75, 90]` |
| Range object | `{"start": 30, "stop": 90, "step": 10}` | `[30, 40, 50, 60, 70, 80, 90]` |

### Using `extra_args`

The `extra_args` field passes additional CLI arguments directly to the
encoder tool. Keys are argument names (without leading dashes) and values
are argument values:

```json
{
  "format": "avif",
  "quality": [60, 75],
  "speed": 4,
  "extra_args": {
    "sharpness": 2,
    "ignore-exif": true
  }
}
```

Note that `extra_args` values are recorded in the quality results for
traceability but are not currently passed to the encoder CLI automatically.
To use them, the encoder method in `src/encoder.py` needs to be updated.
See [Extend formats and metrics](extend-formats-and-metrics) for guidance
on exposing additional encoder parameters.

## Run the study

1. **Ensure the dataset is available:**

   ```bash
   just fetch div2k-valid
   ```

2. **Run the pipeline with a time budget:**

   ```bash
   just pipeline my-study 30m
   ```

3. **Analyze results:**

   ```bash
   just analyze my-study
   ```

4. **Generate visual comparisons:**

   ```bash
   just compare my-study
   ```

5. **Generate a report (includes all studies with results):**

   ```bash
   just report
   ```

## Schema validation

Adding `"$schema": "../study.schema.json"` to your file enables in-editor
validation and autocompletion in VS Code. The schema enforces valid
format names, quality ranges, and parameter bounds.

## Tips

- **Start small**: Use `max_images: 5` and a short time budget (`5m`)
  to verify your config before running a full study.
- **Dry run**: Preview what would run without executing:

  ```bash
  python3 scripts/run_pipeline.py my-study --time-budget 5m --dry-run
  ```

- **Parameter count**: The pipeline runs every combination of parameters.
  With 10 quality levels × 4 speed settings × 2 chroma modes = 80 variants
  per image — plan time budgets accordingly.
- **Clean up**: Remove a study's data with `just clean-study my-study`.

## See also

- [Run the pipeline](run-pipeline) — time budgets, advanced options, output structure
- [Analyze results](analyze-results) — understand CSV and plot outputs
- [Configuration reference](../reference/configuration) — full schema details
- [Extend formats and metrics](extend-formats-and-metrics) — expose additional encoder parameters
