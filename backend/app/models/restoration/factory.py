import logging
from typing import Any

from backend.app.models.restoration.base import BaseFaceRestorer
from backend.app.models.restoration.classic_enhancer import ClassicEnhancer
from backend.app.models.restoration.codeformer_adapter import CodeFormerRestorer
from backend.app.models.restoration.gfpgan_adapter import GFPGANRestorer

logger = logging.getLogger("personaforge.restoration.factory")


class RestorationFactory:
    """
    Factory for instantiating face restoration modules.
    Enforces Safeguard 2: Transparently reports AI availability
    and never conflates classical enhancement with AI restoration.
    """

    @classmethod
    def create_restorer(
        cls,
        name: str = "gfpgan",
        fallback_to_classic: bool = True,
        model_path: str | None = None,
    ) -> tuple[BaseFaceRestorer, dict[str, Any]]:
        """
        Creates the requested face restorer.
        Returns:
            (restorer_instance, transparency_status_report)
        """
        requested = (name or "gfpgan").lower().strip()
        restorer: BaseFaceRestorer | None = None

        if requested in ("gfpgan", "gfpgan_1.4"):
            restorer = GFPGANRestorer(model_path=model_path)
        elif requested in ("codeformer", "code_former"):
            restorer = CodeFormerRestorer(model_path=model_path)
        elif requested in ("classic", "bilateral", "unsharp"):
            restorer = ClassicEnhancer()
        elif requested in ("none", "disabled", "off"):
            restorer = None
        else:
            logger.warning("Unknown restoration adapter '%s'. Defaulting to GFPGAN.", requested)
            restorer = GFPGANRestorer(model_path=model_path)

        # Check availability
        if restorer is not None and restorer.is_available():
            info = restorer.get_model_info()
            status = {
                "ai_restoration": "Available" if info.get("is_ai") else "Unavailable",
                "classic_enhancement": "Disabled" if info.get("is_ai") else "Enabled (Bilateral / Unsharp)",
                "active_restorer": info.get("name"),
                "is_ai": bool(info.get("is_ai")),
                "license": info.get("license"),
                "status_message": f"AI Restoration: {info.get('name')} active ({info.get('license')}).",
            }
            return restorer, status

        # If AI model is unavailable, handle fallback
        if fallback_to_classic and requested not in ("none", "disabled", "off"):
            logger.info(
                "Requested AI restorer '%s' weights unavailable. Falling back to ClassicEnhancer.",
                requested,
            )
            fallback = ClassicEnhancer()
            status = {
                "ai_restoration": "Unavailable",
                "classic_enhancement": "Enabled (Bilateral / Unsharp)",
                "active_restorer": "ClassicEnhancer",
                "is_ai": False,
                "license": "Apache 2.0 (OpenCV Native)",
                "status_message": "AI Restoration: Unavailable. Optional Classic Enhancement: Enabled (Bilateral / Unsharp).",
            }
            return fallback, status

        # Disabled completely
        disabled_enhancer = ClassicEnhancer(unsharp_amount=0.0)
        status = {
            "ai_restoration": "Unavailable",
            "classic_enhancement": "Disabled",
            "active_restorer": "None",
            "is_ai": False,
            "license": "None",
            "status_message": "Face restoration disabled.",
        }
        return disabled_enhancer, status
