from abc import ABC, abstractmethod
import numpy as np

class BaseSwapModel(ABC):
    """
    Adapter interface for face swap models.
    Enforces the Open/Closed Principle for plugging in new architectures.
    """

    @abstractmethod
    def load_model(self, model_path: str, providers: list[str]) -> None:
        """Loads the model into memory."""
        pass

    @abstractmethod
    def swap_face(self, frame: np.ndarray, target_face, source_face) -> np.ndarray:
        """
        Executes the swap.
        :param frame: The full target image/frame.
        :param target_face: The face bounding box/landmarks in the target frame.
        :param source_face: The extracted embedding/landmarks of the source identity.
        :return: The frame with the face swapped.
        """
        pass

    @abstractmethod
    def validate_input(self, target_face, source_face) -> bool:
        """Validates if the provided face objects are compatible with this model."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Frees model resources."""
        pass
