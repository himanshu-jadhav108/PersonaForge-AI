"""
factory.py — Factory function for obtaining face blending implementations.
"""

from pipelines.blending.alpha_blend import AlphaBlend
from pipelines.blending.base import BaseBlender
from pipelines.blending.feathered_blend import FeatheredBlend
from pipelines.blending.seamless_clone import SeamlessCloneExperimental


def get_blender(blender_type: str | None = None) -> BaseBlender:
    """
    Instantiate and return a BaseBlender implementation.

    Parameters
    ----------
    blender_type : Optional[str]
        One of 'alpha', 'feathered', 'seamless_clone_experimental'.
        Defaults to 'feathered' for optimal quality-speed balance.

    Returns
    -------
    BaseBlender
    """
    b_type = (blender_type or "feathered").lower().strip()

    if b_type in ("alpha", "direct", "fast"):
        return AlphaBlend()
    elif b_type in ("seamless_clone", "seamless_clone_experimental", "experimental"):
        return SeamlessCloneExperimental()
    else:
        return FeatheredBlend()
