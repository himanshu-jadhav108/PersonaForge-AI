from enum import Enum

from pydantic import BaseModel, Field


class DriftZone(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftStatus(BaseModel):
    zone: DriftZone
    is_drift: bool = Field(..., description="Whether similarity drops below stable threshold or sudden drop occurs")
    is_sudden_drop: bool = Field(..., description="Sudden drop >= 0.20 relative to rolling window average")
    drop_delta: float = Field(0.0, description="Magnitude of sudden drop from rolling average")


class TimelinePoint(BaseModel):
    frame_index: int
    timestamp_sec: float
    similarity: float
    rolling_similarity: float
    drift_zone: DriftZone
    is_sudden_drop: bool = False
    crop_thumbnail_url: str | None = None


class FrameIdentityRecord(BaseModel):
    frame_index: int
    timestamp: float
    similarity_score: float
    is_drift: bool
    rolling_similarity: float | None = None
    drift_zone: str | None = None
    sudden_drop: bool = False


class IdentityReport(BaseModel):
    job_id: str
    identity_score: float = Field(..., description="Overall identity confidence score (0-100)")
    drift_detected: bool = Field(..., description="Whether any identity drift was detected")
    average_similarity: float = Field(..., description="Average similarity score across all frames")
    min_similarity: float = Field(..., description="Minimum similarity score across all frames")
    max_similarity: float = Field(0.0, description="Maximum similarity score across all frames")
    drift_occurrences: int = Field(..., description="Number of frames where drift was detected")
    total_frames_analyzed: int
    stable_frame_count: int = 0
    warning_frame_count: int = 0
    critical_frame_count: int = 0
    sudden_drops_count: int = 0
    records: list[FrameIdentityRecord] = Field(default_factory=list)
    timeline: list[TimelinePoint] = Field(default_factory=list)
    json_report_path: str | None = None
    html_report_path: str | None = None
