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
from xml.etree.ElementTree import Element, ElementTree, SubElement

from jinja2 import Environment, FileSystemLoader

from src.interactive import figure_to_html_fragment, generate_study_figures
from src.report_images import (
    discover_and_optimise,
    img_srcset_html,
    optimise_svg,
    picture_html,
)
from src.site_config import (
    asset_paths,
    canonical_url,
    copy_deployable_assets,
    get_site_config,
    minify_html_document,
)

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_report_sitemap(output_dir: Path) -> None:
    """Generate sitemap.xml for the report section."""
    site_config = get_site_config()
    urlset = Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for html_file in sorted(output_dir.rglob("*.html")):
        if html_file.name == "404.html":
            continue
        content = html_file.read_text(encoding="utf-8")
        if 'name="robots" content="noindex' in content.lower():
            continue
        relative = html_file.relative_to(output_dir).as_posix()
        if relative == "index.html":
            loc = canonical_url(f"{site_config.report_subpath}/")
        elif relative.endswith("/index.html"):
            loc = canonical_url(f"{site_config.report_subpath}/{relative[: -len('index.html')]}")
        else:
            loc = canonical_url(f"{site_config.report_subpath}/{relative}")
        SubElement(SubElement(urlset, "url"), "loc").text = loc
    ElementTree(urlset).write(output_dir / "sitemap.xml", encoding="utf-8", xml_declaration=True)


REPORT_DIR = PROJECT_ROOT / "report"
TEMPLATES_DIR = REPORT_DIR / "templates"
ASSETS_DIR = REPORT_DIR / "assets"
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "report"

# Plotly CDN URL for the basic bundle
PLOTLY_BASIC_VERSION = "3.3.1"
PLOTLY_BASIC_URL = f"https://cdn.plot.ly/plotly-basic-{PLOTLY_BASIC_VERSION}.min.js"

# PhotoSwipe CDN URLs
PHOTOSWIPE_VERSION = "5.4.4"
_PSW_BASE = f"https://cdn.jsdelivr.net/npm/photoswipe@{PHOTOSWIPE_VERSION}/dist"
PHOTOSWIPE_CSS_URL = f"{_PSW_BASE}/photoswipe.css"
PHOTOSWIPE_JS_URL = f"{_PSW_BASE}/photoswipe.esm.min.js"
PHOTOSWIPE_LIGHTBOX_URL = f"{_PSW_BASE}/photoswipe-lightbox.esm.min.js"


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


def ensure_photoswipe_bundle() -> tuple[Path, Path, Path]:
    """Ensure PhotoSwipe assets exist in assets directory, downloading if needed.

    Returns:
        Tuple of (css_path, js_path, lightbox_js_path)
    """
    css_path = ASSETS_DIR / "photoswipe.css"
    js_path = ASSETS_DIR / "photoswipe.esm.min.js"
    lightbox_path = ASSETS_DIR / "photoswipe-lightbox.esm.min.js"

    if css_path.exists() and js_path.exists() and lightbox_path.exists():
        return css_path, js_path, lightbox_path

    print(f"PhotoSwipe bundle not found, downloading version {PHOTOSWIPE_VERSION}...")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for url, dest in [
            (PHOTOSWIPE_CSS_URL, css_path),
            (PHOTOSWIPE_JS_URL, js_path),
            (PHOTOSWIPE_LIGHTBOX_URL, lightbox_path),
        ]:
            if not dest.exists():
                urlretrieve(url, dest)
                size_kb = dest.stat().st_size / 1024
                print(f"  Downloaded {dest.name} ({size_kb:.1f} KB)")
        return css_path, js_path, lightbox_path
    except Exception as e:
        raise RuntimeError(f"Failed to download PhotoSwipe from {_PSW_BASE}: {e}") from e


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

    study_id = data.get("study_id", "unknown")
    description = ""
    study_config = PROJECT_ROOT / "config" / "studies" / f"{study_id}.json"
    if study_config.exists():
        with open(study_config, encoding="utf-8") as f:
            description = json.load(f).get("description", "")

    return {
        "study_id": study_id,
        "study_name": data.get("study_name", data.get("study_id", "Unknown Study")),
        "description": description,
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
        if y_part == "BITS PER PIXEL":
            y_part = "BPP"
        if x_part == "Bits Per Pixel":
            x_part = "BPP"
        return f"{y_part} vs {x_part}"

    return name.replace("_", " ").title()


def generate_study_page(
    quality_json_path: Path,
    env: Environment,
    output_dir: Path,
    plotly_js_path: str,
    index_path: str,
    timestamp: str,
    photoswipe_css_path: str = "",
    photoswipe_js_path: str = "",
    photoswipe_lightbox_js_path: str = "",
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
    site_config = get_site_config()

    # Resolve study config path for analysis parameter overrides
    study_config_path = PROJECT_ROOT / "config" / "studies" / f"{study_id}.json"
    if not study_config_path.exists():
        study_config_path = None

    print(f"  Generating figures for: {metadata['study_name']}")
    figures = generate_study_figures(
        quality_json_path,
        study_config_path=study_config_path,
    )

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

    # Copy and optimise SVG static figures
    svg_files = []
    if analysis_dir.exists():
        for svg_path in analysis_dir.glob("*.svg"):
            svg_dest = data_output / svg_path.name
            optimise_svg(svg_path, svg_dest)
            svg_files.append(f"data/{study_id}/{svg_path.name}")
        if svg_files:
            print(f"  Optimised {len(svg_files)} SVG files")

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
        page_title=f"{metadata['study_name']} | {site_config.site_name}",
        page_description=(
            str(metadata.get("description"))
            or f"Interactive study results for {metadata['study_name']} on dataset {metadata['dataset_id']}."
        ),
        robots_directive="index,follow",
        theme_color=site_config.brand["accent"],
        canonical_url=canonical_url(f"{site_config.report_subpath}/{metadata['filename']}"),
        og_image_url=canonical_url(f"{site_config.report_subpath}/assets/opengraph.png"),
        og_type="article",
        site_name=site_config.site_name,
        repository_url=site_config.repository_url,
        site_assets=asset_paths(),
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
        photoswipe_css_path=photoswipe_css_path,
        photoswipe_js_path=photoswipe_js_path,
        photoswipe_lightbox_js_path=photoswipe_lightbox_js_path,
        index_path=index_path,
        generation_timestamp=timestamp,
    )

    # Write HTML file
    output_file = output_dir / metadata["filename"]
    output_file.write_text(minify_html_document(html), encoding="utf-8")
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

    # Ensure PhotoSwipe bundle is available
    pswp_css_src, pswp_js_src, pswp_lightbox_src = ensure_photoswipe_bundle()

    # Copy plotly.js bundle to output
    assets_output = output_dir / "assets"
    if assets_output.exists():
        shutil.rmtree(assets_output)
    assets_output.mkdir()
    plotly_dst = assets_output / "plotly-basic.min.js"
    shutil.copy2(plotly_src, plotly_dst)
    copy_deployable_assets(assets_output)

    # Copy PhotoSwipe assets to output
    for src in (pswp_css_src, pswp_js_src, pswp_lightbox_src):
        shutil.copy2(src, assets_output / src.name)

    # Set up Jinja2
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,  # We trust our own HTML fragments
    )

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    plotly_js_path = "assets/plotly-basic.min.js"
    pswp_css_path = "assets/photoswipe.css"
    pswp_js_path = "assets/photoswipe.esm.min.js"
    pswp_lightbox_js_path = "assets/photoswipe-lightbox.esm.min.js"
    index_path = "index.html"

    # Generate study pages
    study_metadata_list = []
    for quality_path in study_paths:
        metadata = generate_study_page(
            quality_path,
            env,
            output_dir,
            plotly_js_path,
            index_path,
            timestamp,
            photoswipe_css_path=pswp_css_path,
            photoswipe_js_path=pswp_js_path,
            photoswipe_lightbox_js_path=pswp_lightbox_js_path,
        )
        study_metadata_list.append(metadata)

    # Generate index page
    template = env.get_template("index.html.j2")
    site_config = get_site_config()
    html = template.render(
        page_title=site_config.site_name,
        page_description=(
            "Interactive benchmarks comparing AVIF, JPEG XL, WebP, and JPEG across study results and downloadable figures."
        ),
        robots_directive="index,follow",
        theme_color=site_config.brand["accent"],
        canonical_url=canonical_url(f"{site_config.report_subpath}/"),
        og_image_url=canonical_url(f"{site_config.report_subpath}/assets/opengraph.png"),
        og_type="website",
        site_name=site_config.site_name,
        repository_url=site_config.repository_url,
        site_assets=asset_paths(),
        studies=study_metadata_list,
        plotly_js_path=plotly_js_path,
        index_path=index_path,
        generation_timestamp=timestamp,
    )
    index_file = output_dir / "index.html"
    index_file.write_text(minify_html_document(html), encoding="utf-8")
    print(f"Index written: {index_file}")

    _write_report_sitemap(output_dir)
    print(f"Sitemap written: {output_dir / 'sitemap.xml'}")


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
