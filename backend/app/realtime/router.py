import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    RTCPeerConnection = None
    RTCSessionDescription = None

try:
    from backend.app.realtime.stream_processor import RealTimeProcessor
    from backend.app.realtime.webrtc import FaceSwapVideoStreamTrack
except ImportError:
    FaceSwapVideoStreamTrack = None
    RealTimeProcessor = None

router = APIRouter(prefix="/realtime", tags=["Realtime"])
logger = logging.getLogger("personaforge.realtime.router")


class OfferSchema(BaseModel):
    sdp: str
    type: str
    source_image_path: str | None = None
    session_id: str | None = None
    model_name: str = "inswapper_128.onnx"


# Global references
pcs = set()
global_processor = None


async def close_peer_connections():
    """Close and clean up all active WebRTC peer connections."""
    if pcs:
        import asyncio

        coros = [pc.close() for pc in list(pcs)]
        await asyncio.gather(*coros, return_exceptions=True)
        pcs.clear()


@router.post("/offer")
async def offer(params: OfferSchema):
    if not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="WebRTC streaming is unavailable because 'aiortc' is not installed."
        )
    global global_processor

    resolved_source = params.source_image_path
    if not resolved_source and params.session_id:
        from pathlib import Path

        session_dir = Path("uploads") / params.session_id
        candidates = list(session_dir.glob("source_face.*"))
        if candidates:
            resolved_source = str(candidates[0])

    if not resolved_source or not os.path.exists(resolved_source):
        raise HTTPException(
            status_code=400,
            detail="Source image not found. Please upload a source face image first.",
        )

    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Initialize Processor if needed
    if global_processor is None or global_processor.swapper._app is None:  # simplified logic
        logger.info("Initializing RealTimeProcessor with source '%s'...", resolved_source)
        global_processor = RealTimeProcessor(resolved_source, params.model_name)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if message == "ping":
                channel.send("pong")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received", track.kind)
        if track.kind == "video":
            local_video = FaceSwapVideoStreamTrack(track, global_processor)
            pc.addTrack(local_video)

    # Handle offer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@router.get("/stats")
async def get_stats():
    global global_processor
    if not global_processor:
        return {"status": "Not running"}
    return global_processor.get_stats()


@router.get("/connections", summary="Get active WebRTC connections count and status")
async def get_active_connections():
    return {
        "active_connections": len(pcs),
        "aiortc_available": AIORTC_AVAILABLE,
        "is_processor_initialized": global_processor is not None,
    }


@router.post("/stop", summary="Stop and close all active WebRTC sessions")
async def stop_all_connections():
    count = len(pcs)
    await close_peer_connections()
    return {"status": "stopped", "closed_connections": count}
