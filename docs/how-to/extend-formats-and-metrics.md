---
title: "Extend formats and metrics"
description: "Add new image formats, quality metrics, or expose additional encoder parameters."
---

This guide covers three types of code-level extensions:

1. [Expose additional encoder parameters](#expose-additional-encoder-parameters) — make existing encoder CLI options configurable through study JSON files
2. [Add a new image format](#add-a-new-image-format) — integrate a new encoder and decoder
3. [Add a new quality metric](#add-a-new-quality-metric) — integrate a new measurement tool

All three follow the existing patterns in the codebase and do not require
architectural changes.

## Expose additional encoder parameters

Each encoder CLI tool (e.g., `avifenc`, `cjxl`, `cwebp`) supports many more
options than are currently exposed in the study configuration. For example,
AVIF's chroma subsampling (`-y` flag) is already exposed, but other options
like `--sharpness`, `--depth`, or `--min`/`--max` quantizer are not.

Exposing a new parameter follows this pattern:

### 1. Add the parameter to the encoder method

In `src/encoder.py`, add a new parameter to the relevant `encode_*()` method
and include it in the CLI command:

```python
def encode_avif(
    self,
    input_path: Path,
    quality: int,
    speed: int = 4,
    chroma_subsampling: str | None = None,
    sharpness: int | None = None,       # ← new parameter
    output_name: str | None = None,
) -> EncodeResult:
    cmd = ["avifenc", "-j", "1", "-s", str(speed), "-q", str(quality)]
    if chroma_subsampling is not None:
        cmd.extend(["-y", chroma_subsampling])
    if sharpness is not None:              # ← pass to CLI
        cmd.extend(["--sharpness", str(sharpness)])
    cmd.extend([str(input_path), str(output_path)])
```

### 2. Add the parameter to the study schema

In `config/study.schema.json`, add the new field to the encoder properties:

```json
"sharpness": {
  "description": "AVIF sharpness setting (0-7). Only applicable to AVIF format.",
  "oneOf": [
    { "type": "integer", "minimum": 0, "maximum": 7 },
    {
      "type": "array",
      "items": { "type": "integer", "minimum": 0, "maximum": 7 },
      "minItems": 1
    }
  ]
}
```

### 3. Add the parameter to `EncoderConfig`

In `src/study.py`, add the field to the `EncoderConfig` dataclass and parse
it in `_parse_encoder_config()`:

```python
@dataclass
class EncoderConfig:
    format: str
    quality: list[int]
    # ... existing fields ...
    sharpness: list[int] | None = None   # ← new field
```

Parse it the same way as `speed` or `effort`:

```python
sharpness_raw = data.get("sharpness")
sharpness: list[int] | None = None
if sharpness_raw is not None:
    sharpness = [sharpness_raw] if isinstance(sharpness_raw, int) else sharpness_raw
```

### 4. Update the pipeline dispatch

In `src/pipeline.py`, update the `_encode_and_measure()` function to accept and
pass the new parameter. Find the `elif fmt == "avif"` branch and add the parameter:

```python
elif fmt == "avif":
    s = speed if speed is not None else 6
    result = encoder.encode_avif(
        source_path, quality, speed=s,
        chroma_subsampling=chroma_subsampling,
        sharpness=sharpness,                    # ← pass through
        output_name=output_name,
    )
```

You also need to update the `_process_image()` function to iterate over
the new parameter and pass it through the call chain, following the pattern
used for `speed`, `effort`, `method`, and `chroma_subsampling`.

### 5. Update the quality record

In `src/quality.py`, add the field to `QualityRecord` so it is recorded in results:

```python
@dataclass
class QualityRecord:
    # ... existing fields ...
    sharpness: int | None = None
```

Also update `QualityResults.save()` to serialize the new field.

### 6. Update analysis (if sweeping)

If the parameter will be swept (multiple values), add it as a candidate
varying parameter in `src/analysis.py` in the `determine_varying_parameters()`
function, and add it as a group column candidate in `src/interactive.py`.

### Summary of touch points

| File | What to change |
|------|----------------|
| `src/encoder.py` | Add parameter to `encode_*()` method |
| `config/study.schema.json` | Add field definition with validation |
| `src/study.py` | Add to `EncoderConfig`, parse in `_parse_encoder_config()` |
| `src/pipeline.py` | Pass through dispatch, iterate in `_process_image()` |
| `src/quality.py` | Add to `QualityRecord`, `QualityResults.save()` |
| `src/analysis.py` | Add to varying parameter candidates (if sweeping) |
| `src/interactive.py` | Add to group column candidates (if sweeping) |
| `tests/` | Add tests for the new parameter |

---

## Add a new image format

Adding a new format (e.g., HEIC, WebP2, or a custom codec) requires changes
across several files, but follows consistent patterns.

### 1. Install the encoder and decoder

Edit `.devcontainer/Dockerfile` to install the CLI tools. Follow the patterns
for existing tools — either `apt-get` for packaged tools or build from source:

```dockerfile
# Example: install from a package
RUN apt-get update && apt-get install -y my-encoder my-decoder

# Example: build from source
ARG MY_CODEC_VERSION=v1.0.0
RUN git clone --depth 1 --branch ${MY_CODEC_VERSION} \
      https://github.com/example/my-codec.git /tmp/my-codec && \
    cd /tmp/my-codec && mkdir build && cd build && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    ninja && ninja install && ldconfig && \
    rm -rf /tmp/my-codec
```

Rebuild the dev container after editing the Dockerfile.

### 2. Add the encoder method

In `src/encoder.py`, add a new `encode_<format>()` method to `ImageEncoder`.
Follow the pattern of existing methods — call the CLI tool via `subprocess`,
force single-threaded mode, return an `EncodeResult`:

```python
def encode_myformat(
    self,
    input_path: Path,
    quality: int,
    output_name: str | None = None,
) -> EncodeResult:
    """Encode image to MyFormat."""
    if output_name is None:
        output_name = input_path.stem
    output_path = self.output_dir / f"{output_name}.myext"

    try:
        cmd = [
            "my-encoder",
            "--quality", str(quality),
            "--threads", "1",
            str(input_path),
            "-o", str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return EncodeResult(
            success=True,
            output_path=output_path,
            file_size=output_path.stat().st_size,
        )
    except subprocess.CalledProcessError as e:
        return EncodeResult(
            success=False, output_path=None, file_size=None,
            error_message=e.stderr.decode() if e.stderr else str(e),
        )
```

Also add a version detection branch to `get_encoder_version()`:

```python
elif encoder == "my-encoder":
    result = subprocess.run(
        ["my-encoder", "--version"], capture_output=True, text=True, timeout=5
    )
    match = re.search(r"v?(\d+\.\d+\.\d+)", result.stdout + result.stderr)
    return match.group(1) if match else "unknown"
```

### 3. Add the pipeline dispatch

In `src/pipeline.py`, find the `_encode_and_measure()` function and add a
new `elif` branch for the format. Look for the existing dispatch block:

```python
if fmt == "jpeg":
    result = encoder.encode_jpeg(...)
elif fmt == "webp":
    ...
elif fmt == "avif":
    ...
elif fmt == "jxl":
    ...
elif fmt == "myformat":                         # ← add here
    result = encoder.encode_myformat(
        source_path, quality, output_name=output_name
    )
else:
    return (_error_record(..., f"Unknown format: {fmt}"), None)
```

### 4. Handle decoding for quality measurement

In `src/quality.py`, update the `to_png()` function if Pillow cannot open
the new format. Add a decode branch before the Pillow fallback:

```python
if image_path.suffix.lower() == ".myext":
    try:
        cmd = ["my-decoder", str(image_path), str(output_path)]
        subprocess.run(cmd, capture_output=True, check=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        msg = f"Failed to decode myext file {image_path}: {e}"
        raise OSError(msg) from e
```

### 5. Update schemas

Add the format string to the `format` enum in three schema files:

- `config/study.schema.json` — the `format` property enum
- `config/encoding-results.schema.json` — the `format` property enum
- `config/quality-results.schema.json` — the `format` property enum

```json
"enum": ["jpeg", "webp", "avif", "jxl", "myformat"]
```

### 6. Add a color for visualizations

In `src/interactive.py`, add an entry to `FORMAT_COLORS`:

```python
FORMAT_COLORS: dict[str, str] = {
    "jpeg": "#e377c2",
    "webp": "#2ca02c",
    "avif": "#1f77b4",
    "jxl": "#ff7f0e",
    "myformat": "#9467bd",  # ← new color
}
```

### 7. Create a study configuration

Create a study JSON that uses the new format, e.g.,
`config/studies/myformat-quality-sweep.json`:

```json
{
  "$schema": "../study.schema.json",
  "id": "myformat-quality-sweep",
  "name": "MyFormat Quality Sweep",
  "dataset": { "id": "div2k-valid", "max_images": 10 },
  "encoders": [
    { "format": "myformat", "quality": { "start": 30, "stop": 90, "step": 10 } }
  ]
}
```

### 8. Add tests

Follow the existing test patterns in `tests/test_encoder.py` for encoding
and `tests/test_pipeline.py` for integration. At minimum, test that the
encoder produces output and returns a valid `EncodeResult`.

### Summary of touch points

| File | What to change |
|------|----------------|
| `.devcontainer/Dockerfile` | Install encoder/decoder CLI tools |
| `src/encoder.py` | Add `encode_<format>()` + `get_encoder_version()` |
| `src/pipeline.py` | Add `elif` dispatch branch |
| `src/quality.py` | Add decode branch in `to_png()` (if needed) |
| `config/study.schema.json` | Add to `format` enum |
| `config/encoding-results.schema.json` | Add to `format` enum |
| `config/quality-results.schema.json` | Add to `format` enum |
| `src/interactive.py` | Add `FORMAT_COLORS` entry |
| `config/studies/` | Create a study config for the new format |
| `tests/` | Add encoder and integration tests |

---

## Add a new quality metric

Adding a new quality metric (e.g., VMAF, LPIPS, DISTS) follows a similar
pattern to adding a format.

### 1. Install the measurement tool

Edit `.devcontainer/Dockerfile` to install the tool:

```dockerfile
# VMAF example (often bundled with FFmpeg)
RUN apt-get update && apt-get install -y libvmaf-dev
```

Rebuild the dev container.

### 2. Add the measurement method

In `src/quality.py`, add a new `measure_<metric>()` method to
`QualityMeasurer`. Follow the pattern of existing methods — call the CLI
tool, parse the output, return `float | None`:

```python
def measure_mymetric(self, original: Path, compressed: Path) -> float | None:
    """Measure MyMetric between original and compressed images."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_png = Path(tmpdir) / "original.png"
            comp_png = Path(tmpdir) / "compressed.png"
            self._to_png(original, orig_png)
            self._to_png(compressed, comp_png)

            result = subprocess.run(
                ["my-metric-tool", str(orig_png), str(comp_png)],
                capture_output=True, text=True, check=True,
            )
            # Parse the score from stdout
            match = re.search(r"score:\s*([\d.]+)", result.stdout)
            return float(match.group(1)) if match else None
    except Exception:
        return None
```

### 3. Update dataclasses

Add the field to `QualityMetrics`:

```python
@dataclass
class QualityMetrics:
    ssimulacra2: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    butteraugli: float | None = None
    mymetric: float | None = None         # ← new field
    error_message: str | None = None
```

Include it in `measure_all()`:

```python
def measure_all(self, original, compressed, distmap_path=None) -> QualityMetrics:
    # ... existing calls ...
    return QualityMetrics(
        ssimulacra2=self.measure_ssimulacra2(original, compressed),
        psnr=self.measure_psnr(original, compressed),
        ssim=self.measure_ssim(original, compressed),
        butteraugli=butteraugli,
        mymetric=self.measure_mymetric(original, compressed),  # ← add
    )
```

Add the field to `QualityRecord`:

```python
@dataclass
class QualityRecord:
    # ... existing fields ...
    mymetric: float | None = None
```

Update `QualityResults.save()` to serialize it (follow the pattern of
existing metric fields in the serialization dict).

### 4. Add version detection

In `src/quality.py`, add a branch to `get_measurement_tool_version()`:

```python
elif tool == "my-metric-tool":
    result = subprocess.run(
        ["my-metric-tool", "--version"], capture_output=True, text=True, timeout=5
    )
    match = re.search(r"v?(\d+\.\d+\.\d+)", result.stdout + result.stderr)
    return match.group(1) if match else "unknown"
```

### 5. Update the pipeline

In `src/pipeline.py`:

- In `_encode_and_measure()`, map the new metric from `QualityMetrics` to
  `QualityRecord` (find where `metrics.ssimulacra2` is mapped).
- In `_error_record()`, add `mymetric=None` to the error record.
- In `_collect_tool_versions()`, add `"my-metric-tool"` to the list of
  tools whose versions are collected.

### 6. Update analysis and visualization

In `src/analysis.py`:

- Add `"mymetric"` to the `metrics` list in `compute_statistics()`.
- Add `"bytes_per_mymetric_per_pixel"` computation in
  `create_analysis_dataframe()`, following the pattern for existing metrics.
- Add direction entry in `METRIC_DIRECTIONS`:

  ```python
  "mymetric": "higher",  # or "lower" if lower is better
  "bytes_per_mymetric_per_pixel": "lower",
  ```

In `src/interactive.py`, add human-readable labels:

```python
METRIC_LABELS: dict[str, str] = {
    # ... existing entries ...
    "mymetric": "MyMetric",
    "bytes_per_mymetric_per_pixel": "Bytes per MyMetric per Pixel",
}
```

### 7. Update the quality results schema

In `config/quality-results.schema.json`, add the new metric field to the
measurement properties:

```json
"mymetric": {
  "type": ["number", "null"],
  "description": "MyMetric score"
}
```

### 8. Add tests

Follow patterns in `tests/test_quality.py` and `tests/test_pipeline.py`.

### Summary of touch points

| File | What to change |
|------|----------------|
| `.devcontainer/Dockerfile` | Install measurement tool |
| `src/quality.py` | Add `measure_<metric>()`, update `QualityMetrics`, `QualityRecord`, `measure_all()`, `save()`, `get_measurement_tool_version()` |
| `src/pipeline.py` | Map metric in `_encode_and_measure()`, `_error_record()`, `_collect_tool_versions()` |
| `src/analysis.py` | Add to `metrics`, `METRIC_DIRECTIONS`, derived metric computation |
| `src/interactive.py` | Add to `METRIC_LABELS` |
| `config/quality-results.schema.json` | Add field definition |
| `tests/` | Add measurement and integration tests |

## General notes

- **Run tests after every change**: `just check` runs linting, type checking,
  and the full test suite.
- **Type safety**: The project uses mypy in strict mode. All new functions
  need type annotations.
- **Single-threaded encoding**: Always force single-threaded mode in encoder
  CLI calls (e.g., `-j 1`, `--num_threads=1`, `--threads 1`). Parallelism is
  handled at the pipeline level.
- **Error handling**: Encoder and measurement methods return `None` or error
  objects rather than raising exceptions. Follow this pattern for robustness.

## See also

- [Architecture](../explanation/architecture) — design decisions and module overview
- [Configuration reference](../reference/configuration) — study schema details
- [Tools reference](../reference/tools) — encoder and measurement tool CLI usage
- [Create a custom study](create-study) — define studies using the new format or metric
