---
title: Build Documentation
description: Generate and publish the project documentation website using Astro Starlight
---

This guide shows how to generate and publish the project documentation website.

## Overview

The documentation is built using [Astro Starlight](https://starlight.astro.build/), combining:

- **Markdown documentation** from `docs/` (following [Diátaxis framework](https://diataxis.fr/))
- **API reference** auto-generated from Python docstrings using [lazydocs](https://github.com/ml-tooling/lazydocs)

## Prerequisites

- Node.js and npm (included in dev container)
- Python development dependencies: `just install-dev`

## Quick Start

### Preview Documentation Locally

```bash
# Generate content and start development server
just docs-dev
```

Visit <http://localhost:4321> — changes to markdown files hot-reload automatically.

### Build for Local Testing

```bash
# Build without GitHub Pages base path
just docs-build-local

# Serve the built site
just docs-serve
```

Visit <http://localhost:8000> to test the production build.

### Build for GitHub Pages

```bash
# Build with GitHub Pages base path
just docs-build
```

The built site in `docs-site/dist/` is ready for deployment.

## Available Commands

| Command | Description |
|---------|-------------|
| `just docs-generate` | Generate docs content from source files |
| `just docs-dev` | Start development server (port 4321) |
| `just docs-build` | Build for GitHub Pages |
| `just docs-build-local` | Build for local testing |
| `just docs-preview` | Preview built site (port 4321) |
| `just docs-serve [PORT]` | Serve built site (default: 8000) |
| `just docs-clean` | Remove generated files |
| `just docs-rebuild` | Clean and rebuild everything |

## Directory Structure

```text
docs/                           # Source documentation (edit here)
├── tutorials/                  # Learning-oriented guides
├── how-to/                     # Task-oriented solutions
├── reference/                  # Technical descriptions
└── explanation/                # Conceptual discussions

docs-site/                      # Astro Starlight project
├── src/content/docs/           # Generated content (auto-created)
│   ├── index.mdx               # From README.md
│   ├── tutorials/              # Copied from docs/
│   ├── how-to/                 # Copied from docs/
│   ├── reference/              # Copied from docs/
│   │   └── api/                # Auto-generated API reference
│   └── explanation/            # Copied from docs/
├── dist/                       # Built static site
├── astro.config.mjs            # Starlight configuration
└── package.json                # Build scripts
```

## How It Works

### 1. Documentation Generation

The `just docs-generate` command runs `scripts/generate_docs.py`:

1. **Copies documentation** from `docs/` to `docs-site/src/content/docs/`
2. **Generates API reference** from Python docstrings in `src/`
3. **Creates index page** from README.md

### 2. Static Site Build

The `just docs-build` command:

1. Runs `docs-generate` to copy latest content
2. Builds optimized static site with Astro
3. Outputs to `docs-site/dist/`

### Build Variants

- **`npm run build`** — For GitHub Pages (uses `/web-image-formats-research` base path)
- **`npm run build:local`** — For local testing (no base path)

The base path is controlled via `GITHUB_PAGES` environment variable in `docs-site/astro.config.mjs`.

## Writing Documentation

### Adding New Pages

1. Create markdown file in appropriate `docs/` subdirectory:
   - `tutorials/` — Step-by-step learning guides
   - `how-to/` — Problem-solving guides
   - `reference/` — Technical specifications
   - `explanation/` — Background concepts

2. Add frontmatter:

   ```markdown
   ---
   title: Your Page Title
   description: Brief description for SEO
   ---

   # Your Page Title

   Content here...
   ```

3. Run `just docs-generate` to copy to Astro

### Linking Between Pages

Use relative paths **without** `.md` extension:

```markdown
<!-- ✅ Correct -->
See [Getting Started](../tutorials/getting-started)

<!-- ❌ Wrong -->
See [Getting Started](../tutorials/getting-started.md)
```

### Adding Code Examples

Use fenced code blocks with language identifiers:

````markdown
```python
def example():
    return "Hello, World!"
```
````

### Adding Images

Place images in `docs/` directory and reference relatively:

```markdown
![Diagram](../assets/diagram.png)
```

## API Reference

The API reference is automatically generated from Python docstrings.

### Format

Use Google-style docstrings:

```python
def encode_image(path: str, quality: int) -> bool:
    """Encode an image with specified quality.

    Args:
        path: Path to input image
        quality: Encoding quality (0-100)

    Returns:
        True if successful, False otherwise

    Raises:
        FileNotFoundError: If image file not found
    """
    pass
```

### Regeneration

The API reference is regenerated each time you run `just docs-generate`.

## Deployment

### GitHub Pages (Automated)

Add GitHub Actions workflow (`.github/workflows/docs.yml`):

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      
      - name: Install dependencies
        run: cd docs-site && npm install
      
      - name: Generate docs
        run: python3 scripts/generate_docs.py
      
      - name: Build
        run: cd docs-site && npm run build
      
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs-site/dist
```

### Manual Deployment

```bash
# 1. Build site
just docs-build

# 2. Deploy dist/ folder to hosting service
# (GitHub Pages, Netlify, Vercel, etc.)
```

## Features

### Performance

- **~2.4MB total size** for entire documentation
- **Zero JavaScript** by default (only where needed)
- **Sub-second page loads** with aggressive caching
- **Optimized images** with lazy loading

### Built-in Features

- ✅ **Full-text search** (Pagefind)
- ✅ **Mobile-responsive** navigation
- ✅ **Dark mode** support
- ✅ **Syntax highlighting** for code blocks
- ✅ **Automatic sitemap** generation
- ✅ **SEO meta tags**

## Troubleshooting

### Build Fails

**Error**: "docs-site directory not found"

```bash
# Ensure docs-site exists
ls -la docs-site/
```

**Error**: "lazydocs not found"

```bash
# Install development dependencies
just install-dev
```

### Assets Not Loading Locally

If CSS/JS don't load when serving with Python HTTP server:

```bash
# Use build:local for testing
just docs-build-local
just docs-serve
```

### Links Broken

Ensure markdown links don't include `.md` extension:

```bash
# Find links with .md extension
grep -r "\.md)" docs/
```

## Related

- [Astro Starlight Documentation](https://starlight.astro.build/)
- [Diátaxis Framework](https://diataxis.fr/)
- [lazydocs on GitHub](https://github.com/ml-tooling/lazydocs)
