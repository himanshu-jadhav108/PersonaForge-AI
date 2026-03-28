"""Shared OpenCV tracker factory used by CPU/GPU pipelines."""

from __future__ import annotations

import logging
import cv2


logger = logging.getLogger("personaforge.tracker_factory")


def make_tracker(log_prefix: str = ""):
    """Try KCF -> CSRT -> MOSSE trackers; return None if unavailable."""
    prefix = f"{log_prefix} " if log_prefix else ""
    for name, fn in [
        ("KCF",        lambda: cv2.TrackerKCF_create()),
        ("CSRT",       lambda: cv2.TrackerCSRT_create()),
        ("MOSSE",      lambda: cv2.legacy.TrackerMOSSE_create()),
        ("KCF-legacy", lambda: cv2.legacy.TrackerKCF_create()),
    ]:
        try:
            tracker = fn()
            logger.debug("%sUsing %s tracker", prefix, name)
            return tracker
        except AttributeError:
            continue

    logger.warning("%sNo OpenCV tracker available; detecting every frame.", prefix)
    return None
