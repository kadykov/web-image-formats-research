# How-to: Run the Full Pipeline

This guide shows how to run the complete image format comparison pipeline.

## Steps

1. **Fetch test datasets**:

   ```bash
   just fetch-datasets
   ```

2. **Encode images to all formats**:

   ```bash
   just encode
   ```

3. **Measure quality metrics**:

   ```bash
   just measure
   ```

4. **Analyze and visualize results**:

   ```bash
   just analyze
   ```

Or run everything at once:

```bash
just pipeline
```

Results will be saved to the `results/` directory:

- `results/encoded/` — Encoded images in all formats
- `results/metrics/` — Quality measurements (JSON/CSV)
- `results/analysis/` — Visualizations and reports
