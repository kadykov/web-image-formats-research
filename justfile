# Development and workflow commands

# Default recipe lists all available commands
default:
    @just --list

# ── Setup ─────────────────────────────────────────────────────────────

# Install production dependencies
install:
    pip3 install -e .

# Install development dependencies
install-dev:
    pip3 install -e ".[dev]"

# ── Quality checks ───────────────────────────────────────────────────

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

# Check code formatting without making changes
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

# ── Study workflow ───────────────────────────────────────────────────
# These mirror the steps in .github/workflows/study.yml

# Fetch a dataset by ID
fetch DATASET_ID:
    python3 scripts/fetch_dataset.py {{DATASET_ID}}

# Run the full encoding and measurement
pipeline STUDY TIME_BUDGET:
    python3 scripts/run_pipeline.py {{STUDY}} --time-budget {{TIME_BUDGET}} --save-worst-image

# Analyze study results and generate plots
analyze STUDY:
    python3 scripts/analyze_study.py {{STUDY}}

# Generate visual comparison for a study
compare STUDY:
    python3 scripts/generate_comparison.py {{STUDY}}

# Generate interactive HTML report for all studies
report:
    python3 scripts/generate_report.py

# Generate release notes from study results
release-notes:
    python3 scripts/generate_release_notes.py

# Prepare release assets (zip + CSV files)
release-assets:
    python3 scripts/prepare_release_assets.py

# Serve report locally for preview
serve-report PORT="8000":
    @echo "Serving report at http://localhost:{{PORT}}"
    python3 -m http.server {{PORT}} --directory data/report

# ── Documentation ────────────────────────────────────────────────────
# These mirror the steps in .github/workflows/docs.yml

# Generate documentation content from Python docstrings and markdown files
docs-generate:
    python3 scripts/generate_docs.py

# Install documentation site dependencies
docs-install:
    cd docs-site && npm install

# Build documentation site for production
docs-build:
    @just docs-generate
    cd docs-site && npm run build

# Start local documentation development server
docs-dev:
    @echo "Starting documentation development server..."
    @echo "Visit http://localhost:4321"
    cd docs-site && npm run dev

# Preview production build locally
docs-preview:
    @echo "Starting documentation preview server..."
    @echo "Visit http://localhost:4321"
    cd docs-site && npm run preview

# ── Utilities ────────────────────────────────────────────────────────

# Verify all encoding and measurement tools are available
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
