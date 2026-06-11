from backend.app.models.base import BaseSwapModel
from backend.app.models.adapters.inswapper import InSwapperAdapter
from backend.app.models.adapters.simswap import SimSwapAdapter
from backend.app.models.adapters.ghost import GhostAdapter

class ModelFactory:
    """
    Factory class to dynamically load the appropriate swap model adapter based on the model name.
    """
    @staticmethod
    def get_model(model_name: str) -> BaseSwapModel:
        name_lower = model_name.lower()
        
        if "inswapper" in name_lower:
            return InSwapperAdapter()
        elif "simswap" in name_lower:
            return SimSwapAdapter()
        elif "ghost" in name_lower:
            return GhostAdapter()
        else:
            # Default fallback to inswapper
            return InSwapperAdapter()
