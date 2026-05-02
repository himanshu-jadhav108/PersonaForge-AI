import numpy as np

from backend.app.quality.models import TemporalStabilityMetrics


class LandmarkStabilityEvaluator:
    """
    Evaluates temporal face alignment and landmark stability across video frames.
    Follows normalized Inter-Ocular Distance (IOD) jitter formulation in docs/metrics.md.
    """

    JITTER_SPIKE_THRESHOLD_IOD: float = 0.18  # Displacements > 0.18 IOD indicate snap/jitter
    MAX_EXPECTED_JITTER_IOD: float = 0.20     # Normalization scale ceiling for score calculation
    SCENE_CUT_THRESHOLD_IOD: float = 1.00     # Displacements > 1.0 IOD represent scene cuts/resets

    @staticmethod
    def compute_iod(landmarks: np.ndarray) -> float:
        """
        Computes the Inter-Ocular Distance (IOD) between the left and right eyes.
        For standard 5-point facial landmarks (InsightFace):
        index 0: right eye, index 1: left eye.
        """
        if landmarks is None or len(landmarks) < 2:
            return 1.0
        p_right = np.asarray(landmarks[0], dtype=np.float32)
        p_left = np.asarray(landmarks[1], dtype=np.float32)
        dist = float(np.linalg.norm(p_left - p_right))
        return max(1.0, dist)

    @classmethod
    def compute_frame_jitter(cls, prev_landmarks: np.ndarray, curr_landmarks: np.ndarray) -> float:
        """
        Computes the mean Euclidean landmark displacement normalized by the current IOD.
        Returns displacement in IOD units.
        """
        if prev_landmarks is None or curr_landmarks is None:
            return 0.0

        p_prev = np.asarray(prev_landmarks, dtype=np.float32)
        p_curr = np.asarray(curr_landmarks, dtype=np.float32)

        if p_prev.shape != p_curr.shape or len(p_prev) == 0:
            return 0.0

        iod = cls.compute_iod(p_curr)
        displacements = np.linalg.norm(p_curr - p_prev, axis=1)
        mean_disp = float(np.mean(displacements))
        return mean_disp / iod

    @classmethod
    def evaluate_sequence(cls, landmarks_sequence: list[np.ndarray | None]) -> TemporalStabilityMetrics:
        """
        Evaluates a temporal sequence of facial landmarks, detecting jitter spikes
        and calculating the composite stability score.
        """
        if not landmarks_sequence or len(landmarks_sequence) < 2:
            return TemporalStabilityMetrics(
                mean_jitter_iod=0.0,
                max_jitter_iod=0.0,
                jitter_spikes_count=0,
                stability_score=100.0,
            )

        jitters: list[float] = []
        spikes = 0
        prev_kps = None

        for kps in landmarks_sequence:
            if kps is None or len(kps) == 0:
                prev_kps = None
                continue

            if prev_kps is not None:
                j = cls.compute_frame_jitter(prev_kps, kps)
                # Ignore scene cuts or sudden transitions
                if j <= cls.SCENE_CUT_THRESHOLD_IOD:
                    jitters.append(j)
                    if j >= cls.JITTER_SPIKE_THRESHOLD_IOD:
                        spikes += 1

            prev_kps = kps

        if not jitters:
            return TemporalStabilityMetrics(
                mean_jitter_iod=0.0,
                max_jitter_iod=0.0,
                jitter_spikes_count=0,
                stability_score=100.0,
            )

        mean_j = float(np.mean(jitters))
        max_j = float(np.max(jitters))

        # Normalized stability score: 0 to 100
        norm_stability = max(0.0, min(1.0, 1.0 - (mean_j / cls.MAX_EXPECTED_JITTER_IOD))) * 100.0

        # Spike penalty: deduct 2 points per jitter spike up to 20 points
        spike_penalty = min(20.0, spikes * 2.0)
        final_score = max(0.0, norm_stability - spike_penalty)

        return TemporalStabilityMetrics(
            mean_jitter_iod=round(mean_j, 4),
            max_jitter_iod=round(max_j, 4),
            jitter_spikes_count=spikes,
            stability_score=round(final_score, 2),
        )
