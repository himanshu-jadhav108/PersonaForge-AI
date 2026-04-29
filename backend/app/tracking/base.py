"""
base.py — Abstract interface for face tracking implementations.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseFaceTracker(ABC):
    """
    Abstract base class for all face trackers in PersonaForge AI.
    Decouples the video processing loops from specific tracker algorithms
    (KCF, CSRT, Optical Flow, or pure detection).
    """

    @abstractmethod
    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """
        Initialize the tracker with a reference frame and target bounding box.

        Parameters
        ----------
        frame : np.ndarray
            Current video frame (H, W, 3) BGR.
        bbox : Tuple[int, int, int, int]
            Bounding box in (x, y, w, h) format.
        """

    @abstractmethod
    def update(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
        """
        Update tracker with the next video frame.

        Parameters
        ----------
        frame : np.ndarray
            New video frame (H, W, 3) BGR.

        Returns
        -------
        Tuple[bool, Optional[Tuple[int, int, int, int]]]
            (success, bbox_xywh). If tracking fails or confidence is low,
            success is False and bbox is None (or caller falls back to detection).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state."""
