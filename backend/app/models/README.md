# PersonaForge AI - Pluggable Model Architecture

This directory implements the Adapter Pattern to dynamically support multiple face-swapping models without tight coupling to the core inference pipelines.

## Architecture

- **`BaseSwapModel` (`base.py`)**: Defines the interface that all models must implement (`load_model`, `swap_face`, `validate_input`, `cleanup`).
- **`ModelFactory` (`factory.py`)**: Dynamically evaluates the requested `model_name` string and returns the corresponding concrete adapter.

## Implementing a New Model

To add support for a new model (e.g., Roop, FaceFusion):

1. Create a new adapter file in `adapters/new_model.py`.
2. Inherit from `BaseSwapModel`.
3. Implement `load_model()` to instantiate the PyTorch/ONNX engine.
4. Implement `swap_face()` to handle the tensor I/O and return the swapped Numpy array.
5. Register the new adapter inside `ModelFactory.get_model()`.

## Current Adapters
- `InSwapperAdapter`: Wraps `insightface.model_zoo` (current default).
- `SimSwapAdapter`: Functional stub awaiting weights.
- `GhostAdapter`: Functional stub awaiting weights.
