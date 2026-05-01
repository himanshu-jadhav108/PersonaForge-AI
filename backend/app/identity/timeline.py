from typing import Any

from backend.app.identity.models import DriftZone, TimelinePoint


class IdentityTimeline:
    """
    Manages scrubbable temporal data points and continuous drift intervals for timeline visualization.
    """

    def __init__(self):
        self.points: list[TimelinePoint] = []

    def add_point(
        self,
        frame_index: int,
        timestamp_sec: float,
        similarity: float,
        rolling_similarity: float,
        drift_zone: DriftZone,
        is_sudden_drop: bool = False,
        crop_thumbnail_url: str | None = None,
    ) -> TimelinePoint:
        point = TimelinePoint(
            frame_index=int(frame_index),
            timestamp_sec=round(float(timestamp_sec), 3),
            similarity=round(float(similarity), 4),
            rolling_similarity=round(float(rolling_similarity), 4),
            drift_zone=drift_zone,
            is_sudden_drop=bool(is_sudden_drop),
            crop_thumbnail_url=crop_thumbnail_url,
        )
        self.points.append(point)
        return point

    def get_points(self) -> list[TimelinePoint]:
        return self.points

    def get_drift_intervals(self, min_consecutive_frames: int = 1) -> list[dict[str, Any]]:
        """
        Identifies contiguous video segments where identity was in WARNING or CRITICAL zones.
        Returns a list of interval dicts for frontend scrubbable navigation.
        """
        intervals: list[dict[str, Any]] = []
        cur_interval: dict[str, Any] | None = None

        for pt in self.points:
            if pt.drift_zone in (DriftZone.WARNING, DriftZone.CRITICAL):
                if cur_interval is None:
                    cur_interval = {
                        "start_frame": pt.frame_index,
                        "end_frame": pt.frame_index,
                        "start_time_sec": pt.timestamp_sec,
                        "end_time_sec": pt.timestamp_sec,
                        "severity": pt.drift_zone.value,
                        "min_similarity": pt.similarity,
                        "frame_count": 1,
                    }
                else:
                    cur_interval["end_frame"] = pt.frame_index
                    cur_interval["end_time_sec"] = pt.timestamp_sec
                    cur_interval["frame_count"] += 1
                    cur_interval["min_similarity"] = min(cur_interval["min_similarity"], pt.similarity)
                    if pt.drift_zone == DriftZone.CRITICAL:
                        cur_interval["severity"] = DriftZone.CRITICAL.value
            else:
                if cur_interval is not None:
                    if cur_interval["frame_count"] >= min_consecutive_frames:
                        intervals.append(cur_interval)
                    cur_interval = None

        if cur_interval is not None and cur_interval["frame_count"] >= min_consecutive_frames:
            intervals.append(cur_interval)

        return intervals

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [p.model_dump() for p in self.points]
