from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    blur: float = Field(..., description="Blur score (0-100). Higher is better (less blurry).")
    brightness: float = Field(..., description="Brightness score (0-100).")
    contrast: float = Field(..., description="Contrast score (0-100).")
    face_angle: float = Field(..., description="Face angle score (0-100). Higher is more frontal.")
    occlusion: float = Field(..., description="Occlusion score (0-100). Higher means less occlusion.")
    face_size: float = Field(..., description="Face size score (0-100). Higher means face takes up a good portion of the image.")
    sharpness: float = Field(..., description="Sharpness score (0-100). Higher is better.")

    # Optional combined lighting for the report
    lighting: float | None = Field(None, description="Combined lighting score (0-100).")
    laplacian_variance: float | None = Field(None, description="Raw variance of Laplacian value.")


class QualityReport(BaseModel):
    quality_score: float = Field(..., description="Overall aggregated quality score (0-100).")
    metrics: QualityMetrics
    recommendations: list[str] = Field(default_factory=list, description="Actionable recommendations based on the scores.")


class TemporalStabilityMetrics(BaseModel):
    mean_jitter_iod: float = Field(0.0, description="Mean landmark jitter normalized by Inter-Ocular Distance (IOD).")
    max_jitter_iod: float = Field(0.0, description="Maximum single-frame jitter observed in IOD units.")
    jitter_spikes_count: int = Field(0, description="Number of frames exceeding high jitter threshold (> 0.18 IOD).")
    stability_score: float = Field(100.0, description="Normalized temporal stability score (0-100). Higher is more stable.")


class VideoQualityReport(BaseModel):
    overall_quality_score: float = Field(..., description="Aggregated video quality score (0-100).")
    mean_metrics: QualityMetrics
    temporal_stability: TemporalStabilityMetrics
    total_frames_analyzed: int
    recommendations: list[str] = Field(default_factory=list, description="Actionable recommendations.")
