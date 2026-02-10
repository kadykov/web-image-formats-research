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
    npx markdownlint-cli2 --fix "**/*.md"

# Check formatting without making changes
format-code-check:
    ruff format --check src/ tests/ scripts/

# Check markdown formatting
format-markdown-check:
    npx markdownlint-cli2 "**/*.md"

# Check formatting for both code and markdown
format-check: format-code-check format-markdown-check

# Format code and markdown
format: format-code format-markdown

# Run all quality checks (format, lint, type check, test)
check: format-check lint typecheck test

# Clean generated files
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

# Clean results and datasets (BE CAREFUL!)
clean-data:
    rm -rf results/encoded/*
    rm -rf results/metrics/*
    rm -rf results/analysis/*

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

# Measure quality metrics (placeholder)
measure:
    @echo "Measuring quality metrics..."
    python3 scripts/measure_quality.py

# Analyze results (placeholder)
analyze:
    @echo "Analyzing results..."
    python3 scripts/analyze_results.py

# Run complete pipeline (datasets must be fetched separately with: just fetch <id>)
pipeline: encode measure analyze
    @echo "Pipeline complete!"
    @echo "Note: Make sure you've fetched datasets first with 'just fetch <dataset-id>'"

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
