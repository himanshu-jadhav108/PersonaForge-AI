# Smart Face Selection Engine

The Smart Face Selection Engine is a subsystem within PersonaForge AI that automatically scans target videos to find and rank all potential target faces, eliminating the need for manual face selection.

## Architecture

This module utilizes InsightFace to extract face embeddings and perform clustering:

1. **`models.py`**: Defines the data models `FaceProfile` and `SelectionReport` and the `SelectionMode` enum.
2. **`engine.py`**: The `SmartFaceSelector` class performs the heavy lifting:
   - Samples frames from the video.
   - Extracts faces and their embeddings.
   - Clusters faces into distinct identities based on cosine similarity of the embeddings (threshold > 0.5).
   - Computes identity-specific metrics: 
     - **Average Area**: Box width × height.
     - **Visibility Duration**: Percentage of sampled frames the face was detected in.
     - **Detection Confidence**: Accuracy of the bounding box.
     - **Speaking Score**: Variance of mouth keypoints normalized by face height, tracking lip movement.
   - Ranks the resulting faces based on the desired mode: `Largest Face`, `Most Visible Face`, `Most Frequent Face`, `Main Speaker`, or `Highest Confidence Face`.
3. **`dashboard.py`**: Generates an interactive Plotly grouped bar chart to visualize how the detected faces compare across the tracked metrics.

## API Endpoints

- **`POST /selection/analyze`**: 
  - Expects a `multipart/form-data` request with a `video` file and an optional `mode` query string.
  - Returns a JSON `SelectionReport` identifying the optimal face ID.
- **`GET /selection/dashboard/{filename}`**: Serves the generated HTML Plotly dashboard.
- **`GET /selection/thumbnails/{filename}`**: Serves the generated representative face crop for the clustered identity.

## Testing

Unit tests for clustering algorithms, metric aggregations, and ranking selections are provided in `tests/test_selection_engine.py`.
