#!/usr/bin/env python3
"""Generate interactive HTML report from study quality measurements.

This script reads quality measurement results for one or more studies
and generates a static HTML website with interactive Plotly figures.
"""

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlretrieve

from jinja2 import Environment, FileSystemLoader

from src.interactive import figure_to_html_fragment, generate_study_figures
from src.report_images import (
    discover_and_optimise,
    img_srcset_html,
    picture_html,
)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "report"
TEMPLATES_DIR = REPORT_DIR / "templates"
ASSETS_DIR = REPORT_DIR / "assets"
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "report"

# Plotly CDN URL for the basic bundle
PLOTLY_BASIC_VERSION = "3.3.1"
PLOTLY_BASIC_URL = f"https://cdn.plot.ly/plotly-basic-{PLOTLY_BASIC_VERSION}.min.js"


def ensure_plotly_bundle() -> Path:
    """Ensure plotly-basic.min.js exists in assets directory, download if needed.

    Returns:
        Path to the plotly bundle file
    """
    plotly_path = ASSETS_DIR / "plotly-basic.min.js"

    if plotly_path.exists():
        return plotly_path

    # Bundle doesn't exist, download it
    print(f"Plotly bundle not found, downloading version {PLOTLY_BASIC_VERSION}...")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urlretrieve(PLOTLY_BASIC_URL, plotly_path)
        file_size_mb = plotly_path.stat().st_size / (1024 * 1024)
        print(f"  Downloaded plotly-basic.min.js ({file_size_mb:.1f} MB)")
        return plotly_path
    except Exception as e:
        raise RuntimeError(f"Failed to download Plotly bundle from {PLOTLY_BASIC_URL}: {e}") from e


def discover_studies() -> list[Path]:
    """Find all studies with quality measurements.

    Returns:
        List of paths to quality.json files
    """
    if not METRICS_DIR.exists():
        return []
    studies = []
    for study_dir in sorted(METRICS_DIR.iterdir()):
        if study_dir.is_dir():
            quality_file = study_dir / "quality.json"
            if quality_file.exists():
                studies.append(quality_file)
    return studies


def _slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _load_study_metadata(quality_json_path: Path) -> dict[str, str | int | list[str]]:
    """Load study metadata from quality.json without full data processing.

    Args:
        quality_json_path: Path to quality.json

    Returns:
        Dictionary with study metadata
    """
    with open(quality_json_path) as f:
        data = json.load(f)

    measurements = data.get("measurements", [])
    formats = sorted({m["format"] for m in measurements if "format" in m})

    return {
        "study_id": data.get("study_id", "unknown"),
        "study_name": data.get("study_name", data.get("study_id", "Unknown Study")),
        "description": "",  # quality.json doesn't carry description; could load from config
        "dataset_id": data.get("dataset", {}).get("id", "unknown"),
        "image_count": data.get("dataset", {}).get("image_count", 0),
        "measurement_count": len(measurements),
        "formats": formats,
        "tool_versions": data.get("tool_versions"),
        "filename": f"{data.get('study_id', 'unknown')}.html",
    }


def _figure_key_to_title(key: str, study_id: str) -> str:
    """Convert a figure key to a human-readable section title.

    Args:
        key: Figure key like 'format-comparison_ssimulacra2_vs_quality'
        study_id: Study ID prefix to strip

    Returns:
        Human-readable title
    """
    # Strip study_id prefix
    name = key
    if name.startswith(study_id + "_"):
        name = name[len(study_id) + 1 :]

    # Split into y_metric and x_param
    parts = name.split("_vs_")
    if len(parts) == 2:
        y_part = parts[0].replace("_", " ").upper()
        x_part = parts[1].replace("_", " ").title()
        return f"{y_part} vs {x_part}"

    return name.replace("_", " ").title()


def generate_study_page(
    quality_json_path: Path,
    env: Environment,
    output_dir: Path,
    plotly_js_path: str,
    index_path: str,
    timestamp: str,
) -> dict[str, str | int | list[str]]:
    """Generate an HTML page for a single study.

    Args:
        quality_json_path: Path to quality.json
        env: Jinja2 environment
        output_dir: Directory to write HTML to
        plotly_js_path: Relative path to plotly.js from output dir
        index_path: Relative path to index.html
        timestamp: Generation timestamp string

    Returns:
        Study metadata dictionary (for index page)
    """
    metadata = _load_study_metadata(quality_json_path)
    study_id = str(metadata["study_id"])

    print(f"  Generating figures for: {metadata['study_name']}")
    figures = generate_study_figures(quality_json_path)

    # Copy CSV and SVG files to output data directory
    data_output = output_dir / "data" / study_id
    data_output.mkdir(parents=True, exist_ok=True)

    # Copy CSV statistics file
    analysis_dir = PROJECT_ROOT / "data" / "analysis" / study_id
    csv_source = analysis_dir / f"{study_id}_statistics.csv"
    csv_relative = None
    if csv_source.exists():
        csv_dest = data_output / f"{study_id}_statistics.csv"
        shutil.copy2(csv_source, csv_dest)
        csv_relative = f"data/{study_id}/{study_id}_statistics.csv"
        print(f"  Copied CSV: {csv_dest}")

    # Copy SVG static figures
    svg_files = []
    if analysis_dir.exists():
        for svg_path in analysis_dir.glob("*.svg"):
            svg_dest = data_output / svg_path.name
            shutil.copy2(svg_path, svg_dest)
            svg_files.append(f"data/{study_id}/{svg_path.name}")
        if svg_files:
            print(f"  Copied {len(svg_files)} SVG files")

    # Build sections for template
    sections = []
    for key, fig in figures.items():
        section_title = _figure_key_to_title(key, study_id)
        section_id = _slugify(section_title)
        html_fragment = figure_to_html_fragment(fig)
        sections.append(
            {
                "id": section_id,
                "title": section_title,
                "html_fragment": html_fragment,
            }
        )

    # Discover and optimise comparison images
    comparison_images = discover_and_optimise(
        ANALYSIS_DIR,
        study_id,
        output_dir,
        output_dir,
    )
    has_comparisons = bool(comparison_images.sets)
    if has_comparisons:
        print(
            f"  Optimised {sum(len(s.fragment_grids) + len(s.distmap_grids) + (1 if s.original_annotated else 0) + (1 if s.distortion_map else 0) for s in comparison_images.sets)} comparison images"
        )

    # Render template
    template = env.get_template("study.html.j2")
    html = template.render(
        study_name=metadata["study_name"],
        study_description=metadata.get("description", ""),
        dataset_id=metadata["dataset_id"],
        image_count=metadata["image_count"],
        measurement_count=metadata["measurement_count"],
        formats=metadata["formats"],
        tool_versions=metadata.get("tool_versions"),
        sections=sections,
        csv_data_file=csv_relative,
        svg_files=svg_files,
        comparison_images=comparison_images if has_comparisons else None,
        img_srcset_html=img_srcset_html,
        picture_html=picture_html,
        plotly_js_path=plotly_js_path,
        index_path=index_path,
        generation_timestamp=timestamp,
    )

    # Write HTML file
    output_file = output_dir / metadata["filename"]
    output_file.write_text(html, encoding="utf-8")
    print(f"  Written: {output_file}")

    return metadata


def generate_report(
    study_paths: list[Path],
    output_dir: Path,
) -> None:
    """Generate the complete interactive HTML report.

    Args:
        study_paths: List of paths to quality.json files
        output_dir: Directory to write the report to
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure plotly bundle is available
    plotly_src = ensure_plotly_bundle()

    # Copy plotly.js bundle to output
    assets_output = output_dir / "assets"
    assets_output.mkdir(exist_ok=True)
    plotly_dst = assets_output / "plotly-basic.min.js"
    shutil.copy2(plotly_src, plotly_dst)

    # Set up Jinja2
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,  # We trust our own HTML fragments
    )

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    plotly_js_path = "assets/plotly-basic.min.js"
    index_path = "index.html"

    # Generate study pages
    study_metadata_list = []
    for quality_path in study_paths:
        metadata = generate_study_page(
            quality_path, env, output_dir, plotly_js_path, index_path, timestamp
        )
        study_metadata_list.append(metadata)

    # Generate index page
    template = env.get_template("index.html.j2")
    html = template.render(
        studies=study_metadata_list,
        plotly_js_path=plotly_js_path,
        index_path=index_path,
        generation_timestamp=timestamp,
    )
    index_file = output_dir / "index.html"
    index_file.write_text(html, encoding="utf-8")
    print(f"Index written: {index_file}")


def main() -> int:
    """Main entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML report from study quality measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for all studies
  python scripts/generate_report.py

  # Generate report for specific studies
  python scripts/generate_report.py format-comparison avif-quality-sweep

  # Generate to custom output directory
  python scripts/generate_report.py --output /tmp/report

  # List studies available for report generation
  python scripts/generate_report.py --list
        """,
    )

    parser.add_argument(
        "studies",
        nargs="*",
        help="Study IDs to include (default: all studies with quality data)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available studies with quality measurements",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        study_paths = discover_studies()
        if not study_paths:
            print("No studies with quality measurements found.")
            print("Run the pipeline first: just pipeline <study-id>")
            return 1

        print("Available studies for report generation:")
        for path in study_paths:
            metadata = _load_study_metadata(path)
            print(
                f"  - {metadata['study_id']}: {metadata['study_name']} "
                f"({metadata['measurement_count']} measurements, "
                f"{', '.join(str(f) for f in metadata['formats'])})"
            )
        return 0

    # Discover or filter studies
    all_study_paths = discover_studies()
    if not all_study_paths:
        print("Error: No studies with quality measurements found.")
        print("Run the pipeline first: just pipeline <study-id>")
        return 1

    if args.studies:
        # Filter to requested studies
        study_paths = []
        available_ids = {p.parent.name: p for p in all_study_paths}
        for study_id in args.studies:
            if study_id in available_ids:
                study_paths.append(available_ids[study_id])
            else:
                print(f"Warning: Study '{study_id}' not found, skipping.")
                print(f"  Available: {', '.join(sorted(available_ids.keys()))}")
        if not study_paths:
            print("Error: No valid studies specified.")
            return 1
    else:
        study_paths = all_study_paths

    print(f"Generating report for {len(study_paths)} studies...")
    print(f"Output directory: {args.output}")
    generate_report(study_paths, args.output)
    print("\nReport generated successfully!")
    print(f"  Open: {args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
