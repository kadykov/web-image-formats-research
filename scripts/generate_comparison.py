#!/usr/bin/env python3
"""Generate visual comparison images for study worst-case encodings.

This script reads quality measurement results, identifies the worst-performing
image and encoding parameters, locates the most degraded image region using
Butteraugli spatial distortion maps, and generates side-by-side comparison
images at 2x zoom for visual inspection.

Usage:
    python3 scripts/generate_comparison.py format-comparison
    python3 scripts/generate_comparison.py data/metrics/avif-quality-sweep/quality.json
    python3 scripts/generate_comparison.py format-comparison --crop-size 96 --zoom 3
    python3 scripts/generate_comparison.py format-comparison --metric butteraugli
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

  # Use Butteraugli metric to find worst case
  python scripts/generate_comparison.py format-comparison --metric butteraugli

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
        default=2,
        help="Zoom factor for crops (default: 2 for 200%%)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="ssimulacra2",
        choices=["ssimulacra2", "psnr", "ssim", "butteraugli"],
        help="Metric used to find the worst case (default: ssimulacra2)",
    )

    parser.add_argument(
        "--max-columns",
        type=int,
        default=6,
        help="Maximum images per row in the comparison grid (default: 6)",
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
        print(f"Run quality measurements first: just measure-study {study_id}")
        return 1

    # Determine output directory
    output_dir = args.output or (project_root / "data" / "analysis" / study_id / "comparison")

    # Create comparison config
    config = ComparisonConfig(
        crop_size=args.crop_size,
        zoom_factor=args.zoom,
        metric=args.metric,
        max_columns=args.max_columns,
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
        print(f"  Worst image: {result.worst_source_image}")
        print(f"  Worst format: {result.worst_format} q{result.worst_quality}")
        print(f"  {config.metric}: {result.worst_metric_value:.2f}")
        print(f"  Region: ({result.region.x}, {result.region.y}) "
              f"{result.region.width}x{result.region.height}")
        print(f"  Output images: {len(result.output_images)}")
        for img in result.output_images:
            print(f"    - {img}")
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
