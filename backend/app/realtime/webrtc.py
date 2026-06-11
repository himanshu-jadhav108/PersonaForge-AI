import asyncio
import logging
from aiortc import VideoStreamTrack
from av import VideoFrame

from backend.app.realtime.stream_processor import RealTimeProcessor

logger = logging.getLogger("personaforge.realtime.webrtc")

class FaceSwapVideoStreamTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames from an incoming track.
    """

    def __init__(self, track: VideoStreamTrack, processor: RealTimeProcessor):
        super().__init__()
        self.track = track
        self.processor = processor
        self._queue = asyncio.Queue(maxsize=1) # Low latency queue, drop frames if backlogged

    async def recv(self):
        frame = await self.track.recv()

        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Process frame asynchronously to avoid blocking the WebRTC loop
        loop = asyncio.get_event_loop()
        try:
            # We use a thread pool for the processing to not block the asyncio loop
            processed_img = await loop.run_in_executor(None, self.processor.process_frame, img)
        except Exception as e:
            logger.error("Error processing frame: %s", e)
            processed_img = img

        # Reconstruct VideoFrame
        new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
