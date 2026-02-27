#!/usr/bin/env python3
"""Prepare release assets from study results.

Creates one zip archive per study containing its metrics and analysis
data, and copies individual CSV statistics files for direct download
as release assets.

Usage:
    python3 scripts/prepare_release_assets.py
    python3 scripts/prepare_release_assets.py --output-dir release-assets
    python3 scripts/prepare_release_assets.py --metrics-dir data/metrics --analysis-dir data/analysis
"""

import argparse
import sys
import zipfile
from pathlib import Path


def _discover_study_ids(metrics_dir: Path, analysis_dir: Path) -> list[str]:
    """Collect unique study IDs from metrics and analysis directories.

    A study ID corresponds to any subdirectory name found in either
    *metrics_dir* or *analysis_dir*.
    """
    ids: set[str] = set()
    for parent in (metrics_dir, analysis_dir):
        if parent.exists():
            for child in parent.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    ids.add(child.name)
    return sorted(ids)


def _create_study_zip(
    study_id: str,
    metrics_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Create a zip archive for a single study.

    Returns the created zip path, or ``None`` if the study contained
    no files to archive.
    """
    zip_path = output_dir / f"{study_id}.zip"
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Add metrics (quality.json files)
        study_metrics = metrics_dir / study_id
        if study_metrics.exists():
            for path in sorted(study_metrics.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    arcname = str(path.relative_to(study_metrics))
                    zf.write(path, f"metrics/{arcname}")
                    file_count += 1

        # Add analysis (CSV, SVG, and comparison images)
        study_analysis = analysis_dir / study_id
        if study_analysis.exists():
            for path in sorted(study_analysis.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    arcname = str(path.relative_to(study_analysis))
                    zf.write(path, f"analysis/{arcname}")
                    file_count += 1

    if file_count == 0:
        zip_path.unlink()
        return None

    return zip_path


def prepare_release_assets(
    metrics_dir: Path,
    analysis_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Create release assets from study results.

    Produces:
    - ``<study-id>.zip`` per study (quality measurements + analysis +
      comparison images)
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

    # ── 1. Create per-study zip archives ──────────────────────────────
    study_ids = _discover_study_ids(metrics_dir, analysis_dir)

    for study_id in study_ids:
        zip_path = _create_study_zip(study_id, metrics_dir, analysis_dir, output_dir)
        if zip_path is not None:
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"Created {zip_path.name} ({size_mb:.1f} MB)")
            created.append(zip_path)

    if not study_ids:
        print("Warning: No studies found to archive.")

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
