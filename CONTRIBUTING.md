# Contributing

Thank you for your interest in contributing to the Web Image Formats Research
project! This document provides guidelines for different types of contributions.

For detailed documentation, visit the
[project documentation site](https://github.kadykov.com/web-image-formats-research/docs/).

## Getting started

1. Fork the repository and clone it locally.
2. Open in VS Code and reopen in the dev container.
3. Verify the setup:

   ```bash
   just verify-tools
   just check
   ```

See the [Getting Started tutorial](docs/tutorials/getting-started.md) for
detailed setup instructions.

## Types of contributions

### Improve the published research

The research results published via GitHub Actions and GitHub Releases
are only as good as the encoder configurations and parameter choices.
Since the project maintainer is not involved in the development of the
encoding tools or quality metrics, contributions that improve the accuracy
and fairness of the published research are especially valuable:

- **Better encoder parameters** — If an encoder is being used with
  suboptimal default settings, suggest or contribute improved configurations.
  For example, adjusting the default speed/effort levels or quality ranges
  to better represent an encoder's capabilities.
- **Updated tool versions** — When new versions of encoders (avifenc, cjxl,
  cwebp) or quality metrics (ssimulacra2, butteraugli) are released, updating
  the pinned versions in the Dockerfile ensures the research stays current.
- **Fairer comparisons** — If a study configuration disadvantages a particular
  format (e.g., comparing formats at mismatched quality scales), suggest
  adjustments to make comparisons more representative.
- **Additional study configurations** — Propose new studies that investigate
  interesting parameter combinations or use cases not currently covered.

These contributions can be as simple as opening an issue with a suggestion
or submitting a pull request with updated configurations. The study configs
in `config/studies/` are plain JSON files — no code changes needed.

### Add or improve datasets

Register new image datasets by editing `config/datasets.json`. This
requires no code changes. See the
[Add a custom dataset](docs/how-to/add-dataset.md) guide.

### Create new studies

Define new encoding experiments by adding JSON files to `config/studies/`.
See the [Create a custom study](docs/how-to/create-study.md) guide.

### Add formats or metrics

Integrate new image formats, encoder parameters, or quality metrics
into the codebase. See the
[Extend formats and metrics](docs/how-to/extend-formats-and-metrics.md)
guide for the complete list of touch points.

### Fix bugs or improve code

Bug fixes, performance improvements, and code quality improvements are
welcome. Please include tests for any changes.

### Improve documentation

Documentation improvements — fixing typos, clarifying instructions,
adding examples — are always welcome. The documentation follows the
[Diátaxis](https://diataxis.fr/) framework:

- **Tutorials** (`docs/tutorials/`) — learning-oriented step-by-step guides
- **How-to guides** (`docs/how-to/`) — task-oriented problem-solving
- **Reference** (`docs/reference/`) — information-oriented technical details
- **Explanation** (`docs/explanation/`) — understanding-oriented background

See [Build documentation](docs/how-to/build-documentation.md) for how to
preview changes locally.

## Development workflow

1. Create a branch from `main`.
2. Make your changes.
3. Run quality checks:

   ```bash
   just check  # lint + typecheck + test
   ```

4. Fix any issues:

   ```bash
   just fix    # auto-fix lint and formatting
   ```

5. Submit a pull request.

## Code standards

| Tool | Purpose | Config |
|------|---------|--------|
| [Ruff](https://github.com/astral-sh/ruff) | Linting and formatting | `pyproject.toml` |
| [mypy](http://mypy-lang.org/) | Static type checking (strict mode) | `pyproject.toml` |
| [pytest](https://pytest.org/) | Testing with coverage | `pyproject.toml` |
| [markdownlint](https://github.com/DavidAnson/markdownlint) | Markdown style | `.markdownlint-cli2.yaml` |

All code must pass `just check` before merging. The CI pipeline runs
the same checks automatically on pull requests.

See [Run CI locally](docs/how-to/run-ci-locally.md) for details on
individual check commands.

## Commit guidelines

- Write clear, descriptive commit messages.
- Keep commits focused — one logical change per commit.
- Reference relevant issues in commit messages when applicable.

## Questions?

Open a [GitHub Issue](https://github.com/kadykov/web-image-formats-research/issues)
for questions, suggestions, or discussion.
