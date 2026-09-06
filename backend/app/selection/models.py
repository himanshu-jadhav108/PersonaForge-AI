from enum import Enum

from pydantic import BaseModel, Field


class SelectionMode(str, Enum):
    LARGEST = "Largest Face"
    MOST_VISIBLE = "Most Visible Face"
    MOST_FREQUENT = "Most Frequent Face"
    MAIN_SPEAKER = "Main Speaker"
    HIGHEST_CONFIDENCE = "Highest Confidence Face"


class VideoMetadata(BaseModel):
    duration: float = Field(0.0, description="Video duration in seconds")
    fps: float = Field(0.0, description="Video frames per second")
    total_frames: int = Field(0, description="Total frame count")
    width: int = Field(0, description="Pixel width")
    height: int = Field(0, description="Pixel height")
    orientation: str = Field("unknown", description="Orientation: landscape, portrait, or square")
    aspect_ratio: float = Field(0.0, description="Aspect ratio (width / height)")
    codec: str = Field("unknown", description="Video codec name")
    file_size_mb: float = Field(0.0, description="File size in megabytes")


class FaceProfile(BaseModel):
    face_id: str
    person_label: str = Field("Detected Person", description="Human-readable identity label, e.g. 'Detected Person 1'")
    sample_count: int = Field(1, description="Number of sampled frames this face appears in")
    average_area: float = Field(..., description="Average bounding box area")
    visibility_duration: float = Field(..., description="Percentage of sampled frames this face appears in")
    detection_confidence: float = Field(..., description="Average detection confidence score (0-100)")
    speaking_score: float = Field(
        ..., description="Variance in mouth keypoints distance (approximating speaking activity)"
    )
    frontal_score: float = Field(0.0, description="Frontal pose and clarity score (0-100)")
    thumbnail_url: str | None = Field(None, description="URL or path to representative crop")
    representative_embedding: list[float] | None = Field(None, description="512D ArcFace representative embedding")


# Alias for detected person profile
DetectedIdentityProfile = FaceProfile


class MediaAnalysisResponse(BaseModel):
    job_id: str
    session_id: str | None = None
    video_metadata: VideoMetadata | None = None
    selected_face_id: str
    selected_person_label: str | None = None
    selection_mode: SelectionMode
    confidence_score: float = Field(
        ..., description="Confidence that this face is the optimal choice under the selected mode"
    )
    profiles: list[FaceProfile]
    detected_identities: list[FaceProfile] = Field(
        default_factory=list, description="List of clustered person identities"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Diagnostic heuristic warnings (lighting, resolution, motion)"
    )
    recommended_mode: str = Field("balanced", description="Recommended processing mode ('fast', 'balanced', 'high')")
    dashboard_url: str | None = None


# Backward-compatible alias for existing endpoints and tests
SelectionReport = MediaAnalysisResponse
