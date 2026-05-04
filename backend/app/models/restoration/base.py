from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseFaceRestorer(ABC):
    """
    Abstract contract for face restoration adapters.
    Executes enhancement strictly on aligned face crops (e.g. 512x512),
    bounding memory consumption to constant O(1) per face.
    """

    @abstractmethod
    def restore_crop(self, crop: np.ndarray, blend_weight: float = 1.0) -> np.ndarray:
        """
        Restores facial details on the given crop.
        :param crop: BGR face crop image.
        :param blend_weight: Blending factor between original (0.0) and restored (1.0).
        :return: Enhanced BGR face crop.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if model weights and runtime dependencies are ready."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Returns metadata regarding model name, license, framework, and AI status."""
