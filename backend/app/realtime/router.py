from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription
import json
import os

from backend.app.realtime.webrtc import FaceSwapVideoStreamTrack
from backend.app.realtime.stream_processor import RealTimeProcessor

router = APIRouter(prefix="/realtime", tags=["Realtime"])
logger = logging.getLogger("personaforge.realtime.router")

class OfferSchema(BaseModel):
    sdp: str
    type: str
    source_image_path: str
    model_name: str = "inswapper_128.onnx"

# Global references
pcs = set()
global_processor = None

@router.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    import asyncio
    await asyncio.gather(*coros)
    pcs.clear()

@router.post("/offer")
async def offer(params: OfferSchema):
    global global_processor
    
    if not os.path.exists(params.source_image_path):
        raise HTTPException(status_code=400, detail="Source image not found.")

    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    pc = RTCPeerConnection()
    pcs.add(pc)

    # Initialize Processor if needed
    if global_processor is None or global_processor.swapper._app is None: # simplified logic
        logger.info("Initializing RealTimeProcessor...")
        global_processor = RealTimeProcessor(params.source_image_path, params.model_name)

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
