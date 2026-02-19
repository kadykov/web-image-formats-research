#!/usr/bin/env python3
"""Prepare release assets from study results.

Creates a zip archive with all study data (metrics + analysis) and
copies individual CSV statistics files for direct download as release
assets.

Usage:
    python3 scripts/prepare_release_assets.py
    python3 scripts/prepare_release_assets.py --output-dir release-assets
    python3 scripts/prepare_release_assets.py --metrics-dir data/metrics --analysis-dir data/analysis
"""

import argparse
import sys
import zipfile
from pathlib import Path


def prepare_release_assets(
    metrics_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Create release assets from study results.

    Produces:
    - ``study-results.zip`` containing all metrics and analysis data
    - Individual ``<study>_statistics.csv`` files

    Args:
        metrics_dir: Directory with per-study quality.json files.
        analysis_dir: Directory with per-study analysis outputs.
        output_dir: Destination for the prepared assets.

    Returns:
        List of created asset file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # ── 1. Create the zip archive ─────────────────────────────────────
    zip_path = output_dir / "study-results.zip"
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Add metrics (quality.json files)
        if metrics_dir.exists():
            for path in sorted(metrics_dir.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    arcname = str(path.relative_to(metrics_dir.parent))
                    zf.write(path, arcname)
                    file_count += 1

        # Add analysis (CSV + SVG files)
        if analysis_dir.exists():
            for path in sorted(analysis_dir.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    arcname = str(path.relative_to(analysis_dir.parent))
                    zf.write(path, arcname)
                    file_count += 1

    if file_count > 0:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Created {zip_path.name} ({size_mb:.1f} MB, {file_count} files)")
        created.append(zip_path)
    else:
        # Remove empty zip
        zip_path.unlink()
        print("Warning: No files found to archive.")

    # ── 2. Copy individual CSV statistics files ───────────────────────
    if analysis_dir.exists():
        for csv_file in sorted(analysis_dir.rglob("*_statistics.csv")):
            dest = output_dir / csv_file.name
            dest.write_bytes(csv_file.read_bytes())
            size_kb = dest.stat().st_size / 1024
            print(f"Copied {csv_file.name} ({size_kb:.1f} KB)")
            created.append(dest)

    return created


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare release assets from study results.",
    )
    parser.add_argument(
        "-m",
        "--metrics-dir",
        type=Path,
        default=Path("data/metrics"),
        help="Directory containing quality measurement results (default: data/metrics)",
    )
    parser.add_argument(
        "-a",
        "--analysis-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Directory containing analysis outputs (default: data/analysis)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("release-assets"),
        help="Output directory for release assets (default: release-assets)",
    )

    args = parser.parse_args()

    if not args.metrics_dir.exists() and not args.analysis_dir.exists():
        print("Error: Neither metrics nor analysis directory found.", file=sys.stderr)
        return 1

    assets = prepare_release_assets(args.metrics_dir, args.analysis_dir, args.output_dir)

    if not assets:
        print("No release assets created.", file=sys.stderr)
        return 1

    print(f"\n{len(assets)} release asset(s) ready in {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
