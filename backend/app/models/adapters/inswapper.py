import logging

import numpy as np

from backend.app.models.base import BaseSwapModel

logger = logging.getLogger("personaforge.adapters.inswapper")


class InSwapperAdapter(BaseSwapModel):
    def __init__(self):
        self._model = None
        self._providers = None

    def load_model(self, model_path: str, providers: list[str]) -> None:
        try:
            import insightface

            self._providers = providers
            self._model = insightface.model_zoo.get_model(model_path, providers=providers)
            logger.info("InSwapperAdapter: Model loaded from %s", model_path)
        except Exception as e:
            logger.error("InSwapperAdapter: Failed to load model: %s", e)
            raise

    def swap_face(self, frame: np.ndarray, target_face, source_face) -> np.ndarray:
        if not self._model:
            raise RuntimeError("InSwapperAdapter: Model is not loaded.")
        return self._model.get(frame, target_face, source_face, paste_back=True)

    def validate_input(self, target_face, source_face) -> bool:
        if target_face is None or source_face is None:
            return False
        # InSwapper requires insightface Face objects containing 'embedding' and 'kps'
        return bool(hasattr(source_face, "embedding") and hasattr(target_face, "kps"))

    def cleanup(self) -> None:
        self._model = None
        logger.info("InSwapperAdapter: Model unloaded.")
