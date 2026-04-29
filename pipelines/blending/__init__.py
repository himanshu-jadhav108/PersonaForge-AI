"""
pipelines.blending — Modular face blending strategies for PersonaForge AI.
"""

from pipelines.blending.alpha_blend import AlphaBlend
from pipelines.blending.base import BaseBlender
from pipelines.blending.factory import get_blender
from pipelines.blending.feathered_blend import FeatheredBlend
from pipelines.blending.seamless_clone import SeamlessCloneExperimental

__all__ = [
    "AlphaBlend",
    "BaseBlender",
    "FeatheredBlend",
    "SeamlessCloneExperimental",
    "get_blender",
]
