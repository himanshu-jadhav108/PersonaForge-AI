# PersonaForge AI - Real-Time Face Swapping

This module provides GPU-accelerated real-time face swapping streamed over WebRTC. It is designed to work with an RTX 4060 class GPU, maintaining 15-30 FPS with sub-150ms latency.

## Architecture

- **`router.py`**: Exposes FastAPI endpoints for WebRTC Session Description Protocol (SDP) negotiation (`/realtime/offer`) and a stats endpoint (`/realtime/stats`).
- **`webrtc.py`**: Defines the `FaceSwapVideoStreamTrack` which intercepts incoming webcam frames from the user's browser via `aiortc` and shuttles them to a background ThreadPool executor to avoid blocking the main asyncio event loop.
- **`stream_processor.py`**: The `RealTimeProcessor` class handles the frame manipulation. It monitors its own execution latency and automatically degrades image quality (e.g. bypassing bilateral enhancement filters) if the `LATENCY_THRESHOLD` (150ms) is exceeded.

## Usage

1. Start the main PersonaForge FastAPI server.
2. Ensure you have installed the WebRTC dependencies: `pip install aiortc av`.
3. From a frontend client, capture the webcam stream and generate an SDP offer.
4. POST the offer to `/realtime/offer` along with the absolute path to your `source_image_path`.
5. Apply the returned SDP answer to your `RTCPeerConnection`.
6. Read the incoming swapped video stream track.

## Performance Note
Running this real-time stream fully occupies the global `FaceSwapper` singleton. Do not attempt to run batch video processing jobs simultaneously while the real-time stream is active, or the backend queue will deadlock.
