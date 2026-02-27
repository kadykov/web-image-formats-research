---
title: "Run CI locally"
description: "Run the same lint, typecheck and test checks locally that the CI pipeline runs."
---

Before pushing, run the same checks that CI performs.

## All checks at once

```bash
just check  # format-check + lint + typecheck + test
```

## Individual checks

```bash
just lint          # Ruff linting
just format-check  # Code and markdown formatting
just typecheck     # mypy type checking
just test          # pytest
```

## Fix issues

```bash
just fix  # Auto-format code/markdown + auto-fix lint issues
```
