"""
PersonaForge Face Restoration Package.
Provides modular GFPGAN (Apache 2.0), CodeFormer (Research),
and transparent Classical Enhancement fallback.
"""

from backend.app.models.restoration.base import BaseFaceRestorer
from backend.app.models.restoration.classic_enhancer import ClassicEnhancer
from backend.app.models.restoration.codeformer_adapter import CodeFormerRestorer
from backend.app.models.restoration.factory import RestorationFactory
from backend.app.models.restoration.gfpgan_adapter import GFPGANRestorer
from backend.app.models.restoration.identity_guard import RestorationIdentityGuard

__all__ = [
    "BaseFaceRestorer",
    "ClassicEnhancer",
    "CodeFormerRestorer",
    "GFPGANRestorer",
    "RestorationFactory",
    "RestorationIdentityGuard",
]
