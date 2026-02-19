---
title: "Run CI locally"
description: "Run the same lint, typecheck and test checks locally that the CI pipeline runs."
---

Before pushing, you can run the same checks that CI performs.

## Quick Check (all at once)

```bash
just check  # Runs lint + typecheck + test
```

## Individual Checks

```bash
just lint          # Ruff linting
just format-check  # Ruff format check
just typecheck     # mypy type checking
just test          # pytest
```

## Fix Issues

```bash
just lint-fix  # Auto-fix lintable issues
just format    # Auto-format code
```

## CI Pipeline Overview

The GitHub Actions workflow runs in this order:

1. **Lint & Type Check** — Fast feedback on code quality (runs on bare Ubuntu)
2. **Build Image** — Builds the dev container and pushes to GHCR (cached)
3. **Test Suite** — Runs all tests inside the dev container image (depends on steps 1 and 2)
4. **Markdown Lint** — Validates markdown formatting (runs in parallel)

The Docker image is built once, pushed to GitHub Container Registry,
and reused by the test job. GitHub Actions cache (`type=gha`) avoids
rebuilding layers that haven't changed.
