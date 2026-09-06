import logging
from typing import Any

import cv2
import numpy as np

from backend.app.identity.extractor import FaceEmbeddingExtractor

logger = logging.getLogger("personaforge.restoration.guard")


class RestorationIdentityGuard:
    """
    Prevents generative face restoration from drifting away from source identity geometry.
    Monitors before/after ArcFace cosine similarity and dynamically throttles
    restoration blend factor when a drop exceeding 0.08 occurs.
    """

    MAX_ALLOWED_SIM_DROP: float = 0.08  # Empirical threshold from docs/restoration-decision.md

    def __init__(self, max_allowed_drop: float = 0.08):
        self.max_allowed_drop = float(max_allowed_drop)

    def guard_restoration(
        self,
        source_embedding: Any,
        pre_restored_crop: np.ndarray,
        restored_crop: np.ndarray,
        face_analysis_app: Any = None,
    ) -> tuple[np.ndarray, bool, float, float]:
        """
        Guards restored crop against identity drift.
        Returns:
            (final_crop, drift_detected, safe_weight, sim_drop)
        """
        if source_embedding is None or pre_restored_crop is None or restored_crop is None or face_analysis_app is None:
            return restored_crop, False, 1.0, 0.0

        try:
            # Extract normalized embeddings from both crops
            emb_pre = FaceEmbeddingExtractor.extract_from_crop(pre_restored_crop, face_analysis_app)
            emb_post = FaceEmbeddingExtractor.extract_from_crop(restored_crop, face_analysis_app)

            if emb_pre is None or emb_post is None:
                return restored_crop, False, 1.0, 0.0

            sim_pre = FaceEmbeddingExtractor.compute_similarity(source_embedding, emb_pre)
            sim_post = FaceEmbeddingExtractor.compute_similarity(source_embedding, emb_post)

            sim_drop = max(0.0, sim_pre - sim_post)

            if sim_drop >= self.max_allowed_drop:
                # Identity drifted: dynamically throttle blend weight
                safe_weight = max(0.20, min(1.0, 1.0 - (sim_drop * 5.0)))
                logger.warning(
                    "Restoration identity drift detected (drop: %.3f >= %.3f). Throttling blend weight to %.2f.",
                    sim_drop,
                    self.max_allowed_drop,
                    safe_weight,
                )
                guarded_crop = cv2.addWeighted(restored_crop, safe_weight, pre_restored_crop, 1.0 - safe_weight, 0)
                return guarded_crop, True, round(safe_weight, 3), round(sim_drop, 4)

            return restored_crop, False, 1.0, round(sim_drop, 4)

        except Exception as err:
            logger.debug("Identity guard evaluation bypassed due to exception: %s", err)
            return restored_crop, False, 1.0, 0.0
