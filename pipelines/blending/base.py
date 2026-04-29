"""
base.py — Base interface for face crop blending algorithms.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseBlender(ABC):
    """Abstract base class for all face patch blending implementations."""

    @abstractmethod
    def blend(
        self,
        frame: np.ndarray,
        crop: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        """
        Blend the swapped face crop back into the destination video frame.

        Parameters
        ----------
        frame : np.ndarray
            The full-resolution background frame (H, W, 3) BGR uint8.
        crop : np.ndarray
            The swapped face crop (crop_h, crop_w, 3) BGR uint8.
        x1, y1, x2, y2 : int
            Bounding box coordinates where the crop belongs on the frame.

        Returns
        -------
        np.ndarray
            Composite frame with crop blended in.
        """
