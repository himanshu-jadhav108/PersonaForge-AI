# PersonaForge Benchmark Framework

This module provides a reproducible, automated benchmarking suite for measuring system performance under various face-swapping workloads.

## Features

- **Automated Synthetic Data Generation**: Generates temporary videos simulating 720p, 1080p, and 4K loads, complete with moving "faces" for the underlying models to process.
- **Resource Tracking**: Monitors peak RAM and VRAM usage in a background thread via `psutil` and `GPUtil`.
- **Quality & Identity Scoring**: Validates the visual fidelity of output frames using the `FaceQualityAssessor` and `IdentityValidator` modules.
- **CSV & JSON Reporting**: Aggregates test runs into a historical CSV log and a `latest_run.json` structure for easy ingestion.
- **Plotly Visualizations**: Generates interactive HTML charts comparing FPS, Processing Time, and Memory usage across different configuration parameters.

## Running the Benchmark

You can trigger the benchmark suite from the project root:

```bash
python -m benchmark.runner
```

## Directory Structure
- `benchmark/reports/`: Contains the `benchmark_history.csv` and `latest_run.json`.
- `benchmark/charts/`: Contains the Plotly `.html` chart outputs.
- `outputs/benchmark_temp/`: Temporary directory created during runs to hold synthetic frames.

## Configuration Matrix
Modify the `configs` list inside `runner.py` to add custom configurations, batch sizes, face counts, and resolution parameters. Note that batching and multi-model implementations are natively mapped here to be forward-compatible with upcoming core updates.
