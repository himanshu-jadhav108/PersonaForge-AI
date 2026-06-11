from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class SelectionMode(str, Enum):
    LARGEST = "Largest Face"
    MOST_VISIBLE = "Most Visible Face"
    MOST_FREQUENT = "Most Frequent Face"
    MAIN_SPEAKER = "Main Speaker"
    HIGHEST_CONFIDENCE = "Highest Confidence Face"

class FaceProfile(BaseModel):
    face_id: str
    average_area: float = Field(..., description="Average bounding box area")
    visibility_duration: float = Field(..., description="Percentage of sampled frames this face appears in")
    detection_confidence: float = Field(..., description="Average detection confidence score")
    speaking_score: float = Field(..., description="Variance in mouth keypoints distance (approximating speaking activity)")
    thumbnail_url: Optional[str] = Field(None, description="URL or path to the representative crop of this face")

class SelectionReport(BaseModel):
    job_id: str
    selected_face_id: str
    selection_mode: SelectionMode
    confidence_score: float = Field(..., description="Confidence that this face is the optimal choice under the selected mode")
    profiles: List[FaceProfile]
    dashboard_url: Optional[str] = None
