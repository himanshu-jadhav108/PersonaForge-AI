# Face Quality Assessment System

The Face Quality Assessment module is a subsystem in PersonaForge AI designed to evaluate source face images and provide a quality breakdown before face swapping. This ensures that the user is providing high-quality source materials, leading to significantly better swap outputs.

## Architecture

This module utilizes both OpenCV (for classical computer vision heuristics) and InsightFace (for deep learning face properties):

1. **`models.py`**: Pydantic models `QualityMetrics` and `QualityReport` define the API schemas.
2. **`assessor.py`**: The `FaceQualityAssessor` class implements the analysis:
   - **Blur**: Computed using the Variance of the Laplacian of the image.
   - **Sharpness**: Computed using the mean magnitude of Sobel derivatives.
   - **Lighting (Brightness/Contrast)**: Derived from the mean and standard deviation of grayscale pixel intensities.
   - **Face Size**: Ratio of face bounding box area to total image area.
   - **Face Angle (Pose)**: Computed from pitch, yaw, and roll deviation from 0.
   - **Occlusion**: Approximated from the InsightFace detection confidence score.
3. **`dashboard.py`**: Generates a visual radar chart using Plotly so the user can easily interpret their scores.

## API Endpoints

- **`POST /quality/assess`**: Expects a `multipart/form-data` request with an `image` file. Returns a JSON response containing the `report` (which includes all individual metrics and an overall score, plus actionable recommendations) and a `dashboard_url` link.
- **`GET /quality/dashboard/{filename}`**: Serves the generated HTML Plotly dashboard.

## Testing

A benchmark dataset and unit test suite are provided in `tests/test_quality_assessor.py`. Tests cover verifying that blurred images receive low blur scores, dark images receive low lighting scores, and that clear images generate comprehensive and accurate quality reports.
