from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class BenchmarkConfig(BaseModel):
    id: str = Field(..., description="Unique identifier for the benchmark run configuration")
    device: str = Field(..., description="'cpu' or 'gpu'")
    resolution: str = Field(..., description="e.g., '720p', '1080p', '4k'")
    face_count: int = Field(..., description="Number of faces in the target video")
    model_name: str = Field(default="inswapper_128.onnx", description="Swap model name")
    batch_size: int = Field(default=1, description="Simulated batch size")
    frames: int = Field(default=30, description="Number of frames processed")

class BenchmarkResult(BaseModel):
    config_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_sec: float
    fps: float
    peak_ram_mb: float
    peak_vram_mb: float
    avg_identity_score: float
    avg_quality_score: float
