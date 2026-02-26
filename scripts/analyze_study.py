#!/usr/bin/env python3
"""Analyze study results and generate plots.

This script reads quality measurement results, computes statistics,
and generates visualizations of quality metrics and encoder efficiency.
"""

import argparse
import sys
from pathlib import Path

from src.analysis import analyze_study


def main() -> int:
    """Main entry point for study analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze study quality measurements and generate plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a study by ID (looks for data/metrics/<study-id>/quality.json)
  python scripts/analyze_study.py avif-quality-sweep

  # Analyze with explicit quality.json path
  python scripts/analyze_study.py data/metrics/avif-quality-sweep/quality.json

  # Specify custom output directory
  python scripts/analyze_study.py avif-quality-sweep --output data/analysis/custom-output
        """,
    )

    parser.add_argument(
        "study",
        nargs="?",
        help="Study ID or path to quality.json file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory for analysis results (default: data/analysis/<study-id>)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available studies with quality measurements",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        metrics_dir = Path("data/metrics")
        if not metrics_dir.exists():
            print("No metrics directory found. Run quality measurements first.")
            return 1

        studies = []
        for study_dir in sorted(metrics_dir.iterdir()):
            if study_dir.is_dir():
                quality_file = study_dir / "quality.json"
                if quality_file.exists():
                    studies.append(study_dir.name)

        if not studies:
            print("No studies with quality measurements found.")
            return 1

        print("Available studies with quality measurements:")
        for study in studies:
            print(f"  - {study}")
        return 0

    # Check if study argument is provided
    if not args.study:
        parser.error("the study argument is required (unless --list is used)")
        return 1

    # Determine quality.json path
    study_input = Path(args.study)
    if study_input.suffix == ".json" and study_input.exists():
        # Direct path to quality.json
        quality_json_path = study_input
        study_id = quality_json_path.parent.name
    else:
        # Study ID provided
        study_id = args.study
        quality_json_path = Path("data/metrics") / study_id / "quality.json"

    # Check if quality.json exists
    if not quality_json_path.exists():
        print(f"Error: Quality measurements not found at: {quality_json_path}")
        print(f"Run the pipeline first: just pipeline {study_id}")
        return 1

    # Determine output directory
    output_dir = args.output or Path("data/analysis") / study_id

    # Run analysis
    print(f"Analyzing study: {study_id}")
    print(f"Quality data: {quality_json_path}")
    print(f"Output directory: {output_dir}")
    print()

    try:
        analyze_study(quality_json_path, output_dir)
        print()
        print("Analysis complete!")
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
