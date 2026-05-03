from enum import Enum

from pydantic import BaseModel, Field


class IntegrityTier(str, Enum):
    EXCELLENT = "EXCELLENT"  # >= 85: Studio grade, pristine match and sharpness
    GOOD = "GOOD"            # 70 - 84: Clean swap, minor lighting or angle variation
    FAIR = "FAIR"            # 50 - 69: Noticeable compression or moderate angle
    DEGRADED = "DEGRADED"    # < 50: Severe blur, landmark jitter, or identity drift


class ComponentScore(BaseModel):
    name: str = Field(..., description="Name of the evaluated metric component")
    raw_value: float = Field(..., description="Raw measured value before normalization")
    raw_unit: str = Field(..., description="Unit of the raw metric (e.g. cosine, variance, IOD)")
    normalized_score: float = Field(..., description="Calibrated score on 0-100 scale")
    weight: float = Field(..., description="Weighting fraction applied in composite score (0.0 - 1.0)")
    weighted_contribution: float = Field(..., description="Points contributed to final score")
    status: str = Field(..., description="Status tier: 'optimal', 'acceptable', or 'warning'")
    explanation: str = Field(..., description="Explainable description of this component score")


class IntegrityBreakdown(BaseModel):
    identity: ComponentScore = Field(..., description="ArcFace cosine similarity preservation")
    sharpness: ComponentScore = Field(..., description="Laplacian edge variance high-frequency focus")
    temporal_stability: ComponentScore = Field(..., description="Landmark jitter relative to Inter-Ocular Distance")
    boundary_coherence: ComponentScore | None = Field(None, description="Seam transition gradient continuity")
    detection_confidence: ComponentScore | None = Field(None, description="Detection model confidence score")


class PersonaForgeIntegrityReport(BaseModel):
    job_id: str
    integrity_score: float = Field(..., description="Overall composite PersonaForge Integrity Score (0-100)")
    tier: IntegrityTier = Field(..., description="Executive classification grade")
    breakdown: IntegrityBreakdown = Field(..., description="Individual component score breakdown")
    badges: list[str] = Field(default_factory=list, description="Performance and diagnostic badges")
    explanations: list[str] = Field(default_factory=list, description="Transparent human-readable score explanations")
    recommendations: list[str] = Field(default_factory=list, description="Actionable optimization suggestions")
