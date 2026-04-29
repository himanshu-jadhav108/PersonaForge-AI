"""
backend.app.tracking — Face tracking abstraction layer for PersonaForge AI.
"""

from backend.app.tracking.base import BaseFaceTracker
from backend.app.tracking.detection_tracker import DetectionOnlyTracker
from backend.app.tracking.factory import get_tracker
from backend.app.tracking.kcf_tracker import KCFTracker

__all__ = [
    "BaseFaceTracker",
    "DetectionOnlyTracker",
    "KCFTracker",
    "get_tracker",
]
