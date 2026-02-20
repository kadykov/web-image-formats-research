#!/usr/bin/env python3
"""Generate release notes from study quality measurement results.

Produces a Markdown document suitable for GitHub Releases that summarizes
all completed studies, including dataset information, tool versions,
per-study descriptions, parameter sweeps, and measurement counts.

Usage:
    python3 scripts/generate_release_notes.py
    python3 scripts/generate_release_notes.py --output release-notes.md
    python3 scripts/generate_release_notes.py --metrics-dir data/metrics
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_FORMAT_DISPLAY = {
    "jpeg": "JPEG",
    "webp": "WebP",
    "avif": "AVIF",
    "jxl": "JPEG XL",
}

_CHROMA_DISPLAY = {
    "444": "4:4:4",
    "422": "4:2:2",
    "420": "4:2:0",
    "400": "4:0:0",
}


def _load_quality_data(quality_json: Path) -> dict:
    """Load and return the full quality.json data."""
    with open(quality_json) as f:
        return json.load(f)


def _load_study_config(study_id: str, configs_dir: Path) -> dict | None:
    """Load a study config JSON file, returning None if not found."""
    cfg_path = configs_dir / f"{study_id}.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return json.load(f)


def _describe_int_seq(values: list[int]) -> str:
    """Describe a sorted list of integers as a compact range string.

    Examples:
        [85]          -> "85"
        [50, 55, ..., 95]  -> "50–95 (step 5)"
        [50, 55, 60]  -> "50–60 (step 5)"
        [0, 2, 4, 6]  -> "0–6 (step 2)"
        [1, 3, 7]     -> "1, 3, 7"
    """
    if not values:
        return ""
    if len(values) == 1:
        return str(values[0])
    sorted_vals = sorted(values)
    steps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
    if len(set(steps)) == 1:
        step = steps[0]
        if step == 1:
            return f"{sorted_vals[0]}–{sorted_vals[-1]}"
        return f"{sorted_vals[0]}–{sorted_vals[-1]} (step {step})"
    return ", ".join(str(v) for v in sorted_vals)


def _describe_quality_spec(spec: int | list | dict) -> str:
    """Describe a quality spec from a study config (int, list, or range object)."""
    if isinstance(spec, int):
        return str(spec)
    if isinstance(spec, list):
        return _describe_int_seq(sorted(spec))
    if isinstance(spec, dict):
        start, stop, step = spec["start"], spec["stop"], spec["step"]
        if step == 1:
            return f"{start}–{stop}"
        return f"{start}–{stop} (step {step})"
    return str(spec)


def _describe_int_or_list(spec: int | list[int]) -> str:
    """Describe a parameter that is either a single int or a list of ints."""
    if isinstance(spec, int):
        return str(spec)
    return _describe_int_seq(sorted(spec))


def _format_encoder_from_config(encoder: dict) -> str:
    """Build a parameter description line for one encoder entry from a study config."""
    fmt = encoder["format"]
    label = _FORMAT_DISPLAY.get(fmt, fmt.upper())
    parts = [f"quality {_describe_quality_spec(encoder['quality'])}"]

    if "speed" in encoder:
        parts.append(f"speed {_describe_int_or_list(encoder['speed'])}")
    if "effort" in encoder:
        parts.append(f"effort {_describe_int_or_list(encoder['effort'])}")
    if "method" in encoder:
        parts.append(f"method {_describe_int_or_list(encoder['method'])}")
    if "chroma_subsampling" in encoder:
        chroma_vals = [_CHROMA_DISPLAY.get(c, c) for c in encoder["chroma_subsampling"]]
        parts.append(f"chroma subsampling {'/'.join(chroma_vals)}")

    return f"- {label}: {', '.join(parts)}"


def _format_encoder_from_measurements(fmt: str, measurements: list[dict]) -> str:
    """Derive a parameter description line for one format from actual measurements."""
    label = _FORMAT_DISPLAY.get(fmt, fmt.upper())
    parts: list[str] = []

    for key, display in [
        ("quality", "quality"),
        ("speed", "speed"),
        ("effort", "effort"),
        ("method", "method"),
    ]:
        vals = sorted({m[key] for m in measurements if m.get(key) is not None})
        if vals:
            parts.append(f"{display} {_describe_int_seq(vals)}")

    chroma_vals = sorted(
        {m["chroma_subsampling"] for m in measurements if m.get("chroma_subsampling")}
    )
    if chroma_vals:
        chroma_display = [_CHROMA_DISPLAY.get(c, c) for c in chroma_vals]
        parts.append(f"chroma subsampling {'/'.join(chroma_display)}")

    resolution_vals = sorted(
        {m["resolution"] for m in measurements if m.get("resolution") is not None}
    )
    if len(resolution_vals) > 1:
        parts.append(f"resolution {_describe_int_seq(resolution_vals)} px")

    return f"- {label}: {', '.join(parts)}" if parts else f"- {label}"


def _format_tool_versions(tool_versions: dict[str, str] | None) -> str:
    """Format tool versions as a Markdown table."""
    if not tool_versions:
        return "_Tool version information not available._"

    lines = ["| Tool | Version |", "| --- | --- |"]
    for tool, version in sorted(tool_versions.items()):
        lines.append(f"| {tool} | {version} |")
    return "\n".join(lines)


def _build_study_section(data: dict, study_config: dict | None) -> list[str]:
    """Build the Markdown lines for a single study section."""
    measurements = data.get("measurements", [])
    successful = [m for m in measurements if m.get("measurement_error") is None]
    failed_count = len(measurements) - len(successful)
    unique_images = len({m["source_image"] for m in measurements if "source_image" in m})

    study_name = data.get("study_name", data.get("study_id", "Unknown"))
    lines: list[str] = []
    lines.append(f"#### {study_name}\n")

    # Description — prefer config, fall back to quality.json field
    description = None
    if study_config:
        description = study_config.get("description")
    if not description:
        description = data.get("description")
    if description:
        lines.append(f"{description}\n")

    # Parameters swept
    lines.append("Parameters swept:\n")
    if study_config and study_config.get("encoders"):
        for encoder in study_config["encoders"]:
            lines.append(_format_encoder_from_config(encoder))
        # Preprocessing / resolution sweep
        resize = study_config.get("preprocessing", {}).get("resize")
        if resize:
            lines.append(f"- Resolution (longest edge): {_describe_int_seq(sorted(resize))} px")
    else:
        # Derive from measurements
        formats = sorted({m["format"] for m in measurements if "format" in m})
        for fmt in formats:
            fmt_ms = [m for m in measurements if m["format"] == fmt]
            lines.append(_format_encoder_from_measurements(fmt, fmt_ms))

    lines.append("")

    # Results summary
    result_parts = [f"{len(successful)} successful measurements across {unique_images} images"]
    if failed_count > 0:
        result_parts.append(f"{failed_count} failed")
    lines.append(f"Results: {', '.join(result_parts)}\n")

    return lines


def generate_release_notes(metrics_dir: Path, configs_dir: Path) -> str:
    """Generate Markdown release notes from all study results.

    Args:
        metrics_dir: Directory containing per-study quality.json files.
        configs_dir: Directory containing study config JSON files.

    Returns:
        Markdown string for the release body.
    """
    studies_data: list[tuple[dict, dict | None]] = []
    tool_versions: dict[str, str] | None = None

    for study_dir in sorted(metrics_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        quality_file = study_dir / "quality.json"
        if not quality_file.exists():
            continue

        data = _load_quality_data(quality_file)
        study_id = data.get("study_id", "")
        study_config = _load_study_config(study_id, configs_dir)
        studies_data.append((data, study_config))

        if tool_versions is None and data.get("tool_versions"):
            tool_versions = data["tool_versions"]

    if not studies_data:
        return "No study results found.\n"

    lines: list[str] = []
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("## Study Results\n")
    lines.append(f"Generated: {timestamp}\n")

    # Dataset section — list unique dataset IDs only (image counts vary per study)
    dataset_ids: set[str] = set()
    for data, _ in studies_data:
        ds_id = data.get("dataset", {}).get("id")
        if ds_id:
            dataset_ids.add(ds_id)
    if dataset_ids:
        lines.append("### Dataset\n")
        for ds_id in sorted(dataset_ids):
            lines.append(f"- **{ds_id}**")
        lines.append("")

    # Tool versions
    lines.append("### Tool Versions\n")
    lines.append(_format_tool_versions(tool_versions))
    lines.append("")

    # Per-study summaries
    lines.append("### Studies\n")
    for data, study_config in studies_data:
        lines.extend(_build_study_section(data, study_config))

    # Asset description
    lines.append("### Release Assets\n")
    lines.append("- **`study-results.zip`** — Complete archive of all study data")
    lines.append("  (quality measurements JSON + analysis CSV + SVG figures)")
    lines.append("- **`*_statistics.csv`** — Per-study aggregated statistics")
    lines.append("  (suitable for spreadsheet analysis or plotting)")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate release notes from study results.",
    )
    parser.add_argument(
        "-m",
        "--metrics-dir",
        type=Path,
        default=Path("data/metrics"),
        help="Directory containing per-study quality.json files (default: data/metrics)",
    )
    parser.add_argument(
        "-a",
        "--analysis-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Directory containing analysis results (default: data/analysis)",
    )
    parser.add_argument(
        "-s",
        "--study-configs-dir",
        type=Path,
        default=Path("config/studies"),
        help="Directory containing study config JSON files (default: config/studies)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    if not args.metrics_dir.exists():
        print(f"Error: Metrics directory not found: {args.metrics_dir}", file=sys.stderr)
        return 1

    notes = generate_release_notes(args.metrics_dir, args.study_configs_dir)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(notes, encoding="utf-8")
        print(f"Release notes written to {args.output}")
    else:
        print(notes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
