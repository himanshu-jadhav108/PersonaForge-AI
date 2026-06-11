import pytest
from backend.app.models.factory import ModelFactory
from backend.app.models.adapters.inswapper import InSwapperAdapter
from backend.app.models.adapters.simswap import SimSwapAdapter
from backend.app.models.adapters.ghost import GhostAdapter
import numpy as np

def test_model_factory_returns_correct_adapters():
    inswapper = ModelFactory.get_model("inswapper_128.onnx")
    assert isinstance(inswapper, InSwapperAdapter)

    simswap = ModelFactory.get_model("simswap_224.onnx")
    assert isinstance(simswap, SimSwapAdapter)

    ghost = ModelFactory.get_model("Ghost_256.onnx")
    assert isinstance(ghost, GhostAdapter)

    fallback = ModelFactory.get_model("unknown_model")
    assert isinstance(fallback, InSwapperAdapter)

def test_simswap_stub_bypasses_swap():
    simswap = ModelFactory.get_model("simswap")
    simswap.load_model("dummy_path", ["CPUExecutionProvider"])
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_frame[0, 0] = [255, 255, 255]
    
    out_frame = simswap.swap_face(dummy_frame, "target_face", "source_face")
    
    assert np.array_equal(dummy_frame, out_frame)
    simswap.cleanup()

def test_ghost_stub_bypasses_swap():
    ghost = ModelFactory.get_model("ghost")
    ghost.load_model("dummy_path", ["CPUExecutionProvider"])
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_frame[0, 0] = [255, 255, 255]
    
    out_frame = ghost.swap_face(dummy_frame, "target_face", "source_face")
    
    assert np.array_equal(dummy_frame, out_frame)
    ghost.cleanup()

def test_validation_logic():
    ghost = ModelFactory.get_model("ghost")
    assert ghost.validate_input(None, None) == False
    assert ghost.validate_input("target", "source") == True
