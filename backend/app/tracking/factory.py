"""
factory.py — Tracker factory for obtaining face tracker instances.
"""


from backend.app.tracking.base import BaseFaceTracker
from backend.app.tracking.detection_tracker import DetectionOnlyTracker
from backend.app.tracking.kcf_tracker import KCFTracker


def get_tracker(tracker_type: str | None = "kcf") -> BaseFaceTracker:
    """
    Instantiate and return a BaseFaceTracker implementation.

    Parameters
    ----------
    tracker_type : Optional[str]
        One of 'kcf', 'detection_only', 'detection', 'none'.
        Defaults to 'kcf'.

    Returns
    -------
    BaseFaceTracker
    """
    t_type = (tracker_type or "kcf").lower().strip()

    if t_type in ("detection_only", "detection", "detect", "none", "off"):
        return DetectionOnlyTracker()
    else:
        return KCFTracker()
