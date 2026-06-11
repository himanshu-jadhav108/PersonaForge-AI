# Identity Consistency Engine

The Identity Consistency Engine is a dedicated module for PersonaForge AI designed to continuously monitor and verify the quality of face swaps across long videos. It detects "identity drift", which happens when the generated face gradually drifts away from the source identity over multiple frames.

## Architecture

This module follows SOLID principles and integrates cleanly into the existing face-swap pipeline:

1. **`models.py`**: Defines Pydantic models (`FrameIdentityRecord` and `IdentityReport`) for structured, type-safe data representation.
2. **`validator.py`**: Contains the `IdentityValidator` class which encapsulates the core business logic.
   - Computes cosine similarity between the InsightFace embedding of the source image and the embedding of the swapped face.
   - Detects drift when similarity drops below a configurable threshold (default `0.80`).
   - Generates aggregated JSON reports containing key metrics: Average Similarity, Minimum Similarity, Overall Confidence Score, and Drift Occurrences.
   - Generates visual charts using Plotly.

## Integration

The validator is instantiated per job in `main.py`'s `_run_pipeline`. It is passed down through `process_video_optimized` into the active hardware pipeline (`process_video_gpu` or `process_video_cpu`).

During the frame processing loop, after the face is swapped onto the frame, the `IdentityValidator` is provided with both the source embedding and the newly generated target embedding. It logs the similarity score for each frame.

When the video processing is complete, `IdentityValidator` generates the JSON report and HTML Plotly chart and saves them to `outputs/reports/`.

## API Endpoints

- **`GET /identity/report/{job_id}`**: Retrieves the JSON report containing the metrics for a given job.
