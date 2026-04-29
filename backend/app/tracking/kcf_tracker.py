"""
kcf_tracker.py — Kernelized Correlation Filter (KCF) face tracker.
"""

import logging

import cv2
import numpy as np

from backend.app.tracking.base import BaseFaceTracker

logger = logging.getLogger("personaforge.tracking.kcf")


class KCFTracker(BaseFaceTracker):
    """
    OpenCV-backed KCF / CSRT face tracker with confidence and boundary validation.
    """

    def __init__(self, tracker_type: str = "KCF"):
        self.tracker_type = tracker_type.upper()
        self._tracker = None
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._is_initialized = False

    def _create_raw_tracker(self):
        """Create OpenCV tracking object with robust fallbacks."""
        creators = [
            lambda: cv2.TrackerKCF_create(),
            lambda: cv2.TrackerCSRT_create(),
            lambda: cv2.legacy.TrackerKCF_create(),
            lambda: cv2.legacy.TrackerMOSSE_create(),
        ]
        for fn in creators:
            try:
                return fn()
            except (AttributeError, cv2.error):
                continue
        return None

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self._tracker = self._create_raw_tracker()
        if self._tracker is None:
            logger.debug("OpenCV tracker could not be instantiated; tracker inactive.")
            self._is_initialized = False
            return

        x, y, w, h = [int(v) for v in bbox]
        # Validate bbox has positive volume and fits within frame
        fh, fw = frame.shape[:2]
        if w <= 4 or h <= 4 or x >= fw or y >= fh:
            self._is_initialized = False
            return

        try:
            self._tracker.init(frame, (x, y, w, h))
            self._last_bbox = (x, y, w, h)
            self._is_initialized = True
        except (cv2.error, RuntimeError, ValueError) as e:
            logger.debug("Tracker initialization failed: %s", e)
            self._is_initialized = False

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
        if not self._is_initialized or self._tracker is None:
            return False, None

        fh, fw = frame.shape[:2]
        try:
            ok, raw_bbox = self._tracker.update(frame)
            if not ok:
                self._is_initialized = False
                return False, None

            x, y, w, h = [int(v) for v in raw_bbox]
            # Sanity check: ensure tracked box is inside frame and has realistic aspect
            if (
                w <= 4 or h <= 4
                or x + w < 0 or y + h < 0
                or x >= fw or y >= fh
                or (w / float(max(1, h)) > 3.0)
                or (h / float(max(1, w)) > 3.0)
            ):
                self._is_initialized = False
                return False, None

            self._last_bbox = (x, y, w, h)
            return True, self._last_bbox
        except (cv2.error, RuntimeError, ValueError) as e:
            logger.debug("Tracker update failed: %s", e)
            self._is_initialized = False
            return False, None

    def reset(self) -> None:
        self._tracker = None
        self._last_bbox = None
        self._is_initialized = False
