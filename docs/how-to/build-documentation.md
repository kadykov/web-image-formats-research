---
title: "Build documentation"
description: "Generate and preview the project documentation website using Astro Starlight."
---

## Preview documentation locally

Generate content from source files and start the development server:

```bash
just docs-dev
```

Visit <http://localhost:4321>. Changes to markdown files hot-reload automatically.

## Build for production

```bash
just docs-build
```

The built site goes to `docs-site/dist/`, ready for deployment.

## Available commands

| Command | Description |
|---------|-------------|
| `just docs-generate` | Generate docs content from `docs/` and Python docstrings |
| `just docs-install` | Install documentation site npm dependencies |
| `just docs-dev` | Start development server (port 4321) |
| `just docs-build` | Generate and build for production |
| `just docs-preview` | Preview production build (port 4321) |

## How it works

The `just docs-generate` command runs `scripts/generate_docs.py`, which:

1. Copies markdown files from `docs/` to `docs-site/src/content/docs/`
2. Generates API reference from Python docstrings in `src/` using lazydocs
3. Creates the index page from `README.md`

The `just docs-build` command runs generation first, then builds the
optimized static site with Astro Starlight.

## Writing documentation

### Adding new pages

1. Create a markdown file in the appropriate `docs/` subdirectory:
   - `tutorials/` — step-by-step learning guides
   - `how-to/` — task-oriented solutions
   - `reference/` — technical specifications
   - `explanation/` — background concepts and design decisions

2. Add frontmatter:

   ```markdown
   ---
   title: Your Page Title
   description: Brief description for SEO
   ---
   ```

3. Run `just docs-generate` to copy to the Astro site.

### Linking between pages

Use relative paths **without** `.md` extension:

```markdown
See [Getting Started](../tutorials/getting-started)
```

## Troubleshooting

- **Build fails with "lazydocs not found"**: Run `just install-dev`
- **Broken links**: Ensure markdown links omit the `.md` extension

## See also

- [Astro Starlight docs](https://starlight.astro.build/)
- [Diátaxis framework](https://diataxis.fr/)
