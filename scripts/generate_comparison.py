#!/usr/bin/env python3
"""Generate visual comparison images using interpolation-based quality matching.

This script reads quality measurement results and comparison targets from
the study configuration, selects the most representative source image via
interpolation-based variance analysis, and generates side-by-side comparison
figures at matched quality or file-size levels.

Usage:
    python3 scripts/generate_comparison.py format-comparison
    python3 scripts/generate_comparison.py data/metrics/avif-quality-sweep/quality.json
    python3 scripts/generate_comparison.py format-comparison --crop-size 96 --zoom 3
    python3 scripts/generate_comparison.py format-comparison --source-image data/preprocessed/img.png
"""

import argparse
import sys
from pathlib import Path

from src.comparison import ComparisonConfig, generate_comparison


def main() -> int:
    """Main entry point for the visual comparison script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Generate visual comparison images for encoding artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare by study ID (uses data/metrics/<study-id>/quality.json)
  python scripts/generate_comparison.py avif-quality-sweep

  # Compare using explicit quality.json path
  python scripts/generate_comparison.py data/metrics/format-comparison/quality.json

  # Use larger crop and 3x zoom
  python scripts/generate_comparison.py format-comparison --crop-size 160 --zoom 3

  # Custom output directory
  python scripts/generate_comparison.py format-comparison --output data/analysis/my-comparison
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
        help="Output directory for comparison images (default: data/analysis/<study-id>/comparison)",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
        help="Size of the crop region in pixels before zoom (default: 128)",
    )

    parser.add_argument(
        "--zoom",
        type=int,
        default=3,
        help="Zoom factor for crops (default: 3 for 300%%)",
    )

    parser.add_argument(
        "--max-columns",
        type=int,
        default=4,
        help="Maximum images per row in the comparison grid (default: 4)",
    )

    parser.add_argument(
        "--tile-parameter",
        type=str,
        choices=[
            "format",
            "quality",
            "chroma_subsampling",
            "speed",
            "effort",
            "method",
            "resolution",
        ],
        help=(
            "Which encoding parameter should vary within each comparison figure "
            "(one tile per value of this parameter). All other varying parameters "
            "produce separate figures. Overrides the study-config setting and the "
            "built-in heuristic. Example: --tile-parameter format"
        ),
    )

    parser.add_argument(
        "--source-image",
        type=str,
        help=(
            "Explicitly specify a source image path (relative to project root) "
            "instead of automatic worst-image detection."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available studies with quality measurements",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        metrics_dir = project_root / "data" / "metrics"
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
        quality_json_path = study_input
        study_id = quality_json_path.parent.name
    elif study_input.suffix == ".json":
        # Try resolving relative to project root
        quality_json_path = project_root / study_input
        study_id = quality_json_path.parent.name
    else:
        study_id = args.study
        quality_json_path = project_root / "data" / "metrics" / study_id / "quality.json"

    if not quality_json_path.exists():
        print(f"Error: Quality measurements not found at: {quality_json_path}")
        print(f"Run quality measurements first: just pipeline {study_id}")
        return 1

    # Determine output directory
    output_dir = args.output or (project_root / "data" / "analysis" / study_id / "comparison")

    # Create comparison config
    config = ComparisonConfig(
        crop_size=args.crop_size,
        zoom_factor=args.zoom,
        max_columns=args.max_columns,
        source_image=args.source_image,
        tile_parameter=args.tile_parameter,
    )

    # Generate comparison
    try:
        result = generate_comparison(
            quality_json_path=quality_json_path,
            output_dir=output_dir,
            project_root=project_root,
            config=config,
        )
        print("\nSummary:")
        for tr in result.target_results:
            print(f"\n  Target: {tr.target_metric}={tr.target_value}")
            print(f"    Source image: {tr.source_image}")
            print(
                f"    Region: ({tr.region.x}, {tr.region.y}) {tr.region.width}x{tr.region.height}"
            )
            print(f"    Interpolated qualities: {tr.interpolated_qualities}")
            print(f"    Output images: {len(tr.output_images)}")
            for img in tr.output_images:
                print(f"      - {img}")
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nHint: Source images from the dataset must be available on disk.",
            file=sys.stderr,
        )
        print(
            "  Fetch the dataset: just fetch <dataset-id>",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error generating comparison: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
