# Development and workflow commands

# Default recipe lists all available commands
default:
    @just --list

# Install production dependencies
install:
    pip3 install -e .

# Install development dependencies
install-dev:
    pip3 install -e ".[dev]"

# Run tests
test:
    pytest tests/ -v

# Run tests with coverage
test-cov:
    pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run type checking
typecheck:
    mypy src/

# Run linting
lint:
    ruff check src/ tests/ scripts/

# Run linting with auto-fix
lint-fix:
    ruff check --fix src/ tests/ scripts/

# Format code
format-code:
    ruff format src/ tests/ scripts/

# Format markdown files
format-markdown:
    cd {{justfile_directory()}} && npx markdownlint-cli2 --fix "**/*.md"

# Check formatting without making changes
format-code-check:
    ruff format --check src/ tests/ scripts/

# Check markdown formatting
format-markdown-check:
    cd {{justfile_directory()}} && npx markdownlint-cli2 "**/*.md"

# Check formatting for both code and markdown
format-check: format-code-check format-markdown-check

# Format code and markdown
format: format-code format-markdown

# Automatically fix formatting and lint issues
fix: format lint-fix

# Run all quality checks (format, lint, type check, test)
check: format-check lint typecheck test

# Clean Python cache and build artifacts
clean:
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    rm -rf htmlcov/
    rm -rf .coverage
    rm -rf src/__pycache__/
    rm -rf tests/__pycache__/
    rm -rf scripts/__pycache__/
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Clean all study data (encoded images, preprocessed, metrics) - preserves datasets
clean-studies:
    @echo "Removing all study data (encoded, preprocessed, metrics)..."
    rm -rf data/encoded/*
    rm -rf data/preprocessed/*
    rm -rf data/metrics/*
    @echo "Study data cleaned. Datasets preserved in data/datasets/"

# Clean data for a specific study
clean-study STUDY_ID:
    @echo "Cleaning data for study: {{STUDY_ID}}"
    rm -rf data/encoded/{{STUDY_ID}}
    rm -rf data/preprocessed/{{STUDY_ID}}
    rm -rf data/metrics/{{STUDY_ID}}
    @echo "Study {{STUDY_ID}} data cleaned."

# Clean all data including datasets (BE CAREFUL!)
clean-all-data:
    @echo "WARNING: This will delete ALL data including datasets!"
    @echo "Press Ctrl+C to cancel, or Enter to continue..."
    @read
    rm -rf data/datasets/*
    rm -rf data/encoded/*
    rm -rf data/preprocessed/*
    rm -rf data/metrics/*
    rm -rf data/analysis/*
    @echo "All data cleaned."

# Fetch a dataset by ID
fetch DATASET_ID:
    python3 scripts/fetch_dataset.py {{DATASET_ID}}

# List all available datasets from configuration
list-available-datasets:
    python3 scripts/fetch_dataset.py --list

# List downloaded datasets
list-datasets:
    python3 scripts/fetch_dataset.py --show-downloaded

# Run encoding pipeline (placeholder)
encode:
    @echo "Running encoding pipeline..."
    python3 scripts/encode_images.py

# Run encoding study from a configuration file
run-study STUDY:
    python3 scripts/encode_images.py {{STUDY}}

# Clean and run encoding study (ensures fresh start)
run-study-clean STUDY:
    @echo "Cleaning existing data for study: {{STUDY}}"
    @just clean-study {{STUDY}}
    @echo "Running study: {{STUDY}}"
    python3 scripts/encode_images.py {{STUDY}}

# Dry-run an encoding study (show what would be done)
dry-run-study STUDY:
    python3 scripts/encode_images.py {{STUDY}} --dry-run

# List available study configurations
list-studies:
    python3 scripts/encode_images.py --list

# Measure quality metrics for encoded images
measure RESULTS_FILE:
    python3 scripts/measure_quality.py {{RESULTS_FILE}}

# Measure quality with custom number of workers
measure-with-workers RESULTS_FILE WORKERS:
    python3 scripts/measure_quality.py {{RESULTS_FILE}} --workers {{WORKERS}}

# Measure quality for a specific study by ID (assumes results.json in data/encoded/<study-id>/)
measure-study STUDY_ID:
    python3 scripts/measure_quality.py data/encoded/{{STUDY_ID}}/results.json

# Analyze study results and generate plots
analyze STUDY:
    python3 scripts/analyze_study.py {{STUDY}}

# Analyze study with custom output directory
analyze-to STUDY OUTPUT_DIR:
    python3 scripts/analyze_study.py {{STUDY}} --output {{OUTPUT_DIR}}

# List studies available for analysis
list-analyzed:
    python3 scripts/analyze_study.py --list

# Visual Comparison

# Generate visual comparison for a study (find worst region, create grid)
compare STUDY:
    python3 scripts/generate_comparison.py {{STUDY}}

# Generate comparison with custom crop size and zoom
compare-with STUDY CROP_SIZE ZOOM:
    python3 scripts/generate_comparison.py {{STUDY}} --crop-size {{CROP_SIZE}} --zoom {{ZOOM}}

# Generate comparison to custom output directory
compare-to STUDY OUTPUT_DIR:
    python3 scripts/generate_comparison.py {{STUDY}} --output {{OUTPUT_DIR}}

# Generate comparison using a specific metric to find worst case
compare-metric STUDY METRIC:
    python3 scripts/generate_comparison.py {{STUDY}} --metric {{METRIC}}

# Generate comparison using an aggregate region strategy (average or variance)
compare-strategy STUDY STRATEGY:
    python3 scripts/generate_comparison.py {{STUDY}} --region-strategy {{STRATEGY}}

# List studies available for visual comparison
list-comparisons:
    python3 scripts/generate_comparison.py --list

# Run merged encode+measure pipeline with time budget
pipeline STUDY TIME_BUDGET:
    python3 scripts/run_pipeline.py {{STUDY}} --time-budget {{TIME_BUDGET}}

# Run pipeline without time budget (process all images)
pipeline-all STUDY:
    python3 scripts/run_pipeline.py {{STUDY}}

# Run pipeline and save encoded artifacts to disk
pipeline-save STUDY TIME_BUDGET:
    python3 scripts/run_pipeline.py {{STUDY}} --time-budget {{TIME_BUDGET}} --save-artifacts

# Run pipeline and save encoded files for the worst-quality image (for visual comparison)
pipeline-compare STUDY TIME_BUDGET:
    python3 scripts/run_pipeline.py {{STUDY}} --time-budget {{TIME_BUDGET}} --save-worst-image
    just compare {{STUDY}}

# Run pipeline then analyze results
pipeline-analyze STUDY TIME_BUDGET:
    python3 scripts/run_pipeline.py {{STUDY}} --time-budget {{TIME_BUDGET}}
    just analyze {{STUDY}}
    @echo "Pipeline + analysis complete!"
    @echo "  Quality metrics: data/metrics/{{STUDY}}/quality.json"
    @echo "  Analysis: data/analysis/{{STUDY}}/"

# Run full pipeline: encode+measure, analyze, starting fresh
pipeline-clean STUDY TIME_BUDGET:
    just clean-study {{STUDY}}
    python3 scripts/run_pipeline.py {{STUDY}} --time-budget {{TIME_BUDGET}}
    @echo "Pipeline complete!"
    @echo "  Quality metrics: data/metrics/{{STUDY}}/quality.json"

# Dry-run the pipeline (preview tasks without executing)
pipeline-dry-run STUDY:
    python3 scripts/run_pipeline.py {{STUDY}} --dry-run

# Run old-style separate pipeline: encode + measure + analyze
pipeline-separate STUDY:
    @echo "Running separate encode → measure → analyze pipeline for: {{STUDY}}"
    just run-study {{STUDY}}
    just measure-study {{STUDY}}
    just analyze {{STUDY}}
    @echo "Pipeline complete!"
    @echo "  Encodings: data/encoded/{{STUDY}}/results.json"
    @echo "  Quality metrics: data/metrics/{{STUDY}}/quality.json"
    @echo "  Analysis: data/analysis/{{STUDY}}/"

# Generate interactive HTML report for all studies
report:
    python3 scripts/generate_report.py

# Generate report for specific studies
report-studies +STUDIES:
    python3 scripts/generate_report.py {{STUDIES}}

# Generate report to custom output directory
report-to OUTPUT_DIR +STUDIES:
    python3 scripts/generate_report.py {{STUDIES}} --output {{OUTPUT_DIR}}

# List studies available for report generation
list-report-studies:
    python3 scripts/generate_report.py --list

# Serve report locally for preview
serve-report PORT="8000":
    @echo "Serving report at http://localhost:{{PORT}}"
    python3 -m http.server {{PORT}} --directory data/report

# Release Scripts

# Generate release notes from study results
release-notes:
    python3 scripts/generate_release_notes.py

# Generate release notes to a file
release-notes-to OUTPUT:
    python3 scripts/generate_release_notes.py --output {{OUTPUT}}

# Prepare release assets (zip + CSV files)
release-assets:
    python3 scripts/prepare_release_assets.py

# Prepare release assets to a custom directory
release-assets-to OUTPUT_DIR:
    python3 scripts/prepare_release_assets.py --output-dir {{OUTPUT_DIR}}

# Verify all tools are available
verify-tools:
    @echo "Checking image encoding tools..."
    @command -v cjpeg >/dev/null 2>&1 && echo "✓ cjpeg (JPEG)" || echo "✗ cjpeg missing"
    @command -v cwebp >/dev/null 2>&1 && echo "✓ cwebp (WebP)" || echo "✗ cwebp missing"
    @command -v avifenc >/dev/null 2>&1 && echo "✓ avifenc (AVIF)" || echo "✗ avifenc missing"
    @command -v cjxl >/dev/null 2>&1 && echo "✓ cjxl (JPEG XL)" || echo "✗ cjxl missing"
    @echo ""
    @echo "Checking quality measurement tools..."
    @command -v ssimulacra2 >/dev/null 2>&1 && echo "✓ ssimulacra2" || echo "✗ ssimulacra2 missing"
    @command -v butteraugli_main >/dev/null 2>&1 && echo "✓ butteraugli_main" || echo "✗ butteraugli_main missing"
    @command -v ffmpeg >/dev/null 2>&1 && echo "✓ ffmpeg (PSNR/SSIM)" || echo "✗ ffmpeg missing"
    @echo ""
    @echo "Checking Python packages..."
    @python3 -c "import PIL" 2>/dev/null && echo "✓ Pillow" || echo "✗ Pillow missing"
    @python3 -c "import numpy" 2>/dev/null && echo "✓ NumPy" || echo "✗ NumPy missing"
    @python3 -c "import pandas" 2>/dev/null && echo "✓ Pandas" || echo "✗ Pandas missing"
    @python3 -c "import matplotlib" 2>/dev/null && echo "✓ Matplotlib" || echo "✗ Matplotlib missing"

# Documentation Commands

# Generate documentation content from Python docstrings and markdown files
docs-generate:
    python3 scripts/generate_docs.py

# Install documentation site dependencies
docs-install:
    cd docs-site && npm install

# Start local documentation development server
docs-dev:
    @echo "Starting documentation development server..."
    @echo "Visit http://localhost:4321"
    cd docs-site && npm run dev

# Build documentation site for production (with GitHub Pages path)
docs-build:
    @just docs-generate
    cd docs-site && npm run build

# Build documentation site for local testing (without base path)
docs-build-local:
    @just docs-generate
    cd docs-site && npm run build:local

# Preview production build locally
docs-preview:
    @echo "Starting documentation preview server..."
    @echo "Visit http://localhost:4321"
    cd docs-site && npm run preview

# Build and preview documentation
docs-build-preview: docs-build docs-preview

# Clean documentation build artifacts
docs-clean:
    rm -rf docs-site/dist/
    rm -rf docs-site/.astro/
    rm -rf docs-site/src/content/docs/*
    @echo "Documentation build artifacts cleaned"

# Full documentation workflow: regenerate and build
docs-rebuild: docs-clean docs-generate docs-build

# Serve documentation from dist directory (useful for testing production build)
docs-serve PORT="8000":
    @echo "Serving documentation at http://localhost:{{PORT}}"
    @test -d docs-site/dist || (echo "❌ No build found. Run 'just docs-build-local' first." && exit 1)
    python3 -m http.server {{PORT}} --directory docs-site/dist
