import unittest
import numpy as np

from backend.app.models.factory import ModelFactory
from backend.app.models.adapters.inswapper import InSwapperAdapter
from backend.app.models.adapters.simswap import SimSwapAdapter
from backend.app.models.adapters.ghost import GhostAdapter


class TestModelAdapters(unittest.TestCase):

    def test_model_factory_returns_correct_adapters(self):
        inswapper = ModelFactory.get_model("inswapper_128.onnx")
        self.assertIsInstance(inswapper, InSwapperAdapter)

        simswap = ModelFactory.get_model("simswap_224.onnx")
        self.assertIsInstance(simswap, SimSwapAdapter)

        ghost = ModelFactory.get_model("Ghost_256.onnx")
        self.assertIsInstance(ghost, GhostAdapter)

        fallback = ModelFactory.get_model("unknown_model")
        self.assertIsInstance(fallback, InSwapperAdapter)

    def test_simswap_stub_raises_not_implemented(self):
        simswap = ModelFactory.get_model("simswap")
        simswap.load_model("dummy_path", ["CPUExecutionProvider"])
        
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_frame[0, 0] = [255, 255, 255]
        
        with self.assertRaises(NotImplementedError):
            simswap.swap_face(dummy_frame, "target_face", "source_face")
        simswap.cleanup()

    def test_ghost_stub_raises_not_implemented(self):
        ghost = ModelFactory.get_model("ghost")
        ghost.load_model("dummy_path", ["CPUExecutionProvider"])
        
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_frame[0, 0] = [255, 255, 255]
        
        with self.assertRaises(NotImplementedError):
            ghost.swap_face(dummy_frame, "target_face", "source_face")
        ghost.cleanup()

    def test_validation_logic(self):
        ghost = ModelFactory.get_model("ghost")
        self.assertFalse(ghost.validate_input(None, None))
        self.assertTrue(ghost.validate_input("target", "source"))


if __name__ == "__main__":
    unittest.main()
