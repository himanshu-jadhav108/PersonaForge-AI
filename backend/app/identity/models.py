from pydantic import BaseModel, Field
from typing import List, Optional

class FrameIdentityRecord(BaseModel):
    frame_index: int
    timestamp: float
    similarity_score: float
    is_drift: bool

class IdentityReport(BaseModel):
    job_id: str
    identity_score: float = Field(..., description="Overall identity confidence score (0-100)")
    drift_detected: bool = Field(..., description="Whether any identity drift was detected")
    average_similarity: float = Field(..., description="Average similarity score across all frames")
    min_similarity: float = Field(..., description="Minimum similarity score across all frames")
    drift_occurrences: int = Field(..., description="Number of frames where drift was detected")
    total_frames_analyzed: int
    records: List[FrameIdentityRecord] = Field(default_factory=list)
