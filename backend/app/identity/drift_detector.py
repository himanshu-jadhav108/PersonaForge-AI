import logging

from backend.app.identity.models import DriftStatus, DriftZone

logger = logging.getLogger("personaforge.identity.drift_detector")


class IdentityDriftDetector:
    """
    Evaluates identity stability, drift zones, and sudden drops using calibrated empirical thresholds.
    """

    def __init__(
        self,
        stable_threshold: float = 0.80,
        warning_threshold: float = 0.68,
        critical_threshold: float = 0.55,
        delta_drop_threshold: float = 0.20,
        rolling_window_size: int = 5,
    ):
        self.stable_threshold = float(stable_threshold)
        self.warning_threshold = float(warning_threshold)
        self.critical_threshold = float(critical_threshold)
        self.delta_drop_threshold = float(delta_drop_threshold)
        self.rolling_window_size = int(rolling_window_size)

    def evaluate(self, current_similarity: float, rolling_similarity: float) -> DriftStatus:
        """
        Classifies similarity into STABLE, WARNING, or CRITICAL drift zones,
        and detects sudden drops relative to the moving window average.
        """
        cur = float(current_similarity)
        roll = float(rolling_similarity)

        # 1. Classify drift zone
        if cur >= self.stable_threshold:
            zone = DriftZone.STABLE
        elif cur >= self.warning_threshold:
            zone = DriftZone.WARNING
        else:
            zone = DriftZone.CRITICAL

        # 2. Check for sudden drop across rolling window
        drop_delta = max(0.0, roll - cur)
        is_sudden_drop = bool(drop_delta >= self.delta_drop_threshold)

        # Overall drift flag
        is_drift = bool(cur < self.stable_threshold or is_sudden_drop)

        return DriftStatus(
            zone=zone,
            is_drift=is_drift,
            is_sudden_drop=is_sudden_drop,
            drop_delta=round(drop_delta, 4),
        )

    def is_drift(self, similarity: float) -> bool:
        """
        Fast threshold check for backward compatibility.
        """
        return float(similarity) < self.stable_threshold
