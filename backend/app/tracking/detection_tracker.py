"""
detection_tracker.py — Detection-only face tracker (periodic / per-frame detection).
"""


import numpy as np

from backend.app.tracking.base import BaseFaceTracker


class DetectionOnlyTracker(BaseFaceTracker):
    """
    Tracker implementation that bypasses correlation filters entirely and forces
    the pipeline to rely strictly on periodic or per-frame face detection.
    Useful for high-speed camera cuts, scene changes, or when correlation tracking
    is disabled for maximum accuracy.
    """

    def __init__(self):
        self._last_bbox: tuple[int, int, int, int] | None = None

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self._last_bbox = tuple(int(v) for v in bbox)

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
        # Always returns False to instruct caller to perform face detection
        return False, None

    def reset(self) -> None:
        self._last_bbox = None
