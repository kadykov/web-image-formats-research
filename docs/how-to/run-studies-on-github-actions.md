---
title: "Run studies on GitHub Actions"
description: "Trigger encoding studies on GitHub Actions, monitor progress, and access published results."
---

## Trigger a study run

Studies are run on-demand using GitHub's `workflow_dispatch` trigger.

1. Go to the repository's **Actions** tab.
2. Select the **Run Studies** workflow in the left sidebar.
3. Click **Run workflow**.
4. Fill in the inputs:

   | Input | Default | Description |
   |-------|---------|-------------|
   | **Time budget** | `2h` | Time budget per study (e.g., `30m`, `1h`, `2h`). |
   | **Studies** | `all` | Comma-separated study IDs, or `all` for every study in `config/studies/`. |

5. Click the green **Run workflow** button.

### Example: run specific studies

To run only the format comparison and AVIF speed sweep with a 1-hour budget:

- **Time budget**: `1h`
- **Studies**: `format-comparison, avif-speed-sweep`

## Monitor progress

Click into the workflow run to see the job graph:

1. **Build Container Image** — builds the dev container (~5–10 min, cached after first run)
2. **Prepare Study Matrix** — computes which studies to run
3. **Fetch Dataset** — downloads `div2k-valid` (~2 min)
4. **Study: \<name\>** — one job per study, running in parallel.
   Each study job runs the pipeline, analysis, and comparison generation.
5. **Generate Report** — combines all results into an interactive report
6. **Deploy Report** — publishes to GitHub Pages
7. **Create Release** — tags the commit and creates a GitHub Release

Each job's logs are visible in real time. Study jobs show encoding progress
including image count and time remaining.

## Access results

### Interactive report

After the workflow completes, the report is available at:

```text
https://<username>.github.io/<repository>/report/
```

The report includes interactive Plotly visualizations for all completed studies.

### GitHub Release

Each successful run creates a GitHub Release with:

- **Release notes** — summary of studies, tool versions, and key findings
- **CSV files** — per-study statistics for independent re-analysis

Find releases under the repository's **Releases** tab. The release tag
follows the pattern `study-YYYYMMDD-HHMMSS`.

### Workflow artifacts

Raw data is available as workflow artifacts (90-day retention):

- **study-results-\<id\>** — metrics JSON and analysis outputs per study
- **report** — the generated HTML report
- **release-materials** — release notes and CSV assets

Download artifacts from the workflow run's **Summary** page.

## Run studies in a fork

To run studies in your own fork:

1. **Enable GitHub Actions**: Go to the fork's **Actions** tab and enable
   workflows.

2. **Configure GitHub Pages**: Go to **Settings → Pages → Source**, select
   **Deploy from a branch**, choose `gh-pages` branch with `/ (root)` folder.

3. **Allow GHCR access**: The workflow pushes a container image to GitHub
   Container Registry. This works automatically for public forks.

4. **Trigger the workflow**: Follow the same steps as above.

The report will be deployed to your fork's GitHub Pages URL.

## Customize studies for CI

You can add or modify study configs before triggering a run:

1. Create or edit a study JSON in `config/studies/` (see [Create a custom study](create-study)).
2. Commit and push to your fork.
3. Trigger the workflow with your study ID.

The workflow picks up all study configs from `config/studies/` when
`studies` is set to `all`, or you can specify your custom study ID directly.

## Resource considerations

| Constraint | Limit | Impact |
|------------|-------|--------|
| Job timeout | 6 hours | Limits per-study time budget |
| Runner disk | 14 GB SSD | Limits concurrent encoded artifacts |
| Runner CPU | 4 cores | Affects encoding speed (quality metrics unaffected) |
| Runner RAM | 16 GB | Sufficient for all current studies |
| Concurrent jobs | 20 | Up to 20 studies can run in parallel |
| Artifact retention | 90 days | Raw metrics available for 90 days |

**Tip**: For the DIV2K validation dataset (100 images), a 2-hour budget
typically processes 20–60 images depending on the number of encoder
configurations. Increase the budget for more comprehensive results, up
to the 6-hour job limit.

## See also

- [Public research with GitHub Actions](../explanation/github-actions-research) — why and how this infrastructure works
- [Create a custom study](create-study) — define new studies to run on CI
- [Run the pipeline](run-pipeline) — run studies locally instead
