import numpy as np
import logging
from backend.app.models.base import BaseSwapModel

logger = logging.getLogger("personaforge.adapters.ghost")

class GhostAdapter(BaseSwapModel):
    def __init__(self):
        self._model = None

    def load_model(self, model_path: str, providers: list[str]) -> None:
        logger.info("GhostAdapter: Initializing stub (weights not yet available at %s)", model_path)
        self._model = "stub_loaded"

    def swap_face(self, frame: np.ndarray, target_face, source_face) -> np.ndarray:
        if not self._model:
            raise RuntimeError("GhostAdapter: Model is not loaded.")
        logger.warning("GhostAdapter: swap_face called on stub. Bypassing swap.")
        return frame.copy()

    def validate_input(self, target_face, source_face) -> bool:
        return target_face is not None and source_face is not None

    def cleanup(self) -> None:
        self._model = None
        logger.info("GhostAdapter: Model unloaded.")
