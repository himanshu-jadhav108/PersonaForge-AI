import logging

import numpy as np

from backend.app.models.base import BaseSwapModel

logger = logging.getLogger("personaforge.adapters.simswap")


class SimSwapAdapter(BaseSwapModel):
    def __init__(self):
        self._model = None

    def load_model(self, model_path: str, providers: list[str]) -> None:
        logger.info("SimSwapAdapter: Initializing stub (weights not yet available at %s)", model_path)
        self._model = "stub_loaded"

    def swap_face(self, frame: np.ndarray, target_face, source_face) -> np.ndarray:
        raise NotImplementedError(
            "SimSwapAdapter is a stub and is not yet implemented. "
            "Please use the standard 'inswapper' model for active face swapping."
        )

    def validate_input(self, target_face, source_face) -> bool:
        return target_face is not None and source_face is not None

    def cleanup(self) -> None:
        self._model = None
        logger.info("SimSwapAdapter: Model unloaded.")
