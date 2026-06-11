from pydantic import BaseModel, Field
from typing import List, Optional

class QualityMetrics(BaseModel):
    blur: float = Field(..., description="Blur score (0-100). Higher is better (less blurry).")
    brightness: float = Field(..., description="Brightness score (0-100).")
    contrast: float = Field(..., description="Contrast score (0-100).")
    face_angle: float = Field(..., description="Face angle score (0-100). Higher is more frontal.")
    occlusion: float = Field(..., description="Occlusion score (0-100). Higher means less occlusion.")
    face_size: float = Field(..., description="Face size score (0-100). Higher means face takes up a good portion of the image.")
    sharpness: float = Field(..., description="Sharpness score (0-100). Higher is better.")
    
    # Optional combined lighting for the report
    lighting: Optional[float] = Field(None, description="Combined lighting score (0-100).")

class QualityReport(BaseModel):
    quality_score: float = Field(..., description="Overall aggregated quality score (0-100).")
    metrics: QualityMetrics
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations based on the scores.")
