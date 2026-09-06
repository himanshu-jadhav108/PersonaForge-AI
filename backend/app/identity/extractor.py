import logging
from typing import Any

import numpy as np

logger = logging.getLogger("personaforge.identity.extractor")


class FaceEmbeddingExtractor:
    """
    Handles extraction, normalization, and cosine distance comparison of ArcFace embeddings.
    """

    @staticmethod
    def normalize(embedding: Any) -> np.ndarray | None:
        """
        Normalizes an embedding vector to unit length (L2 norm = 1.0).
        """
        if embedding is None:
            return None
        arr = np.asarray(embedding, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(arr))
        if norm < 1e-6:
            return None
        return arr / norm

    @classmethod
    def extract_from_face(cls, face: Any) -> np.ndarray | None:
        """
        Extracts and normalizes embedding from an InsightFace face object.
        """
        raw = getattr(face, "embedding", None)
        if raw is None:
            return None
        return cls.normalize(raw)

    @classmethod
    def extract_from_crop(cls, crop: np.ndarray, app: Any) -> np.ndarray | None:
        """
        Detects face in crop image and extracts the normalized embedding.
        """
        if app is None or crop is None or crop.size == 0:
            return None
        try:
            faces = app.get(crop)
            if not faces:
                return None
            # Choose the largest detected face
            faces.sort(
                key=lambda f: (
                    (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    if getattr(f, "bbox", None) is not None and len(f.bbox) >= 4
                    else 0
                ),
                reverse=True,
            )
            return cls.extract_from_face(faces[0])
        except Exception as e:
            logger.debug("Failed to extract face embedding from crop: %s", e)
            return None

    @classmethod
    def compute_similarity(cls, source_emb: Any, target_emb: Any) -> float:
        """
        Computes cosine similarity between two face embeddings.
        Returns float between -1.0 and 1.0.
        """
        if source_emb is None or target_emb is None:
            return 0.0

        s_norm = cls.normalize(source_emb)
        t_norm = cls.normalize(target_emb)

        if s_norm is None or t_norm is None:
            return 0.0

        sim = float(np.dot(s_norm, t_norm))
        return round(float(np.clip(sim, -1.0, 1.0)), 4)
