
import numpy as np


class IdentityScorer:
    """
    Computes calibrated statistical metrics and the composite PersonaForge Identity Preservation Score.
    Follows normalization standards defined in docs/metrics.md.
    """

    BASELINE_THRESHOLD: float = 0.20
    HIGH_MATCH_THRESHOLD: float = 0.70

    @classmethod
    def compute_rolling_averages(cls, similarities: list[float], window_size: int = 5) -> list[float]:
        """
        Computes rolling moving average across a trailing window of frames.
        """
        if not similarities:
            return []

        rolling: list[float] = []
        w = max(1, int(window_size))

        for i in range(len(similarities)):
            start_idx = max(0, i - w + 1)
            window = similarities[start_idx : i + 1]
            rolling.append(round(float(np.mean(window)), 4))

        return rolling

    @classmethod
    def compute_statistics(cls, similarities: list[float]) -> dict[str, float]:
        """
        Computes summary statistics: mean, minimum, maximum, and standard deviation.
        """
        if not similarities:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        arr = np.asarray(similarities, dtype=np.float32)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "std": round(float(np.std(arr)), 4),
        }

    @classmethod
    def calculate_identity_score(
        cls,
        similarities: list[float],
        critical_count: int = 0,
        sudden_drops: int = 0,
    ) -> float:
        """
        Computes the normalized Identity Score (0-100) using the formula from docs/metrics.md:
        NormSim = clip((S_cos - baseline) / (high - baseline), 0.0, 1.0) * 100
        with calibrated deductions for sudden drops and critical drift occurrences.
        """
        if not similarities:
            return 0.0

        avg_sim = float(np.mean(similarities))

        # Raw normalized similarity (0 to 100)
        norm_range = cls.HIGH_MATCH_THRESHOLD - cls.BASELINE_THRESHOLD
        raw_norm = float(np.clip((avg_sim - cls.BASELINE_THRESHOLD) / norm_range, 0.0, 1.0)) * 100.0

        # Stability penalty: sudden drops indicate abrupt landmark divergence
        penalty = min(20.0, (float(sudden_drops) * 2.5) + (float(critical_count) * 1.0))

        score = max(0.0, min(100.0, raw_norm - penalty))
        return round(score, 2)
