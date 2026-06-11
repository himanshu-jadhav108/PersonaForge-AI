import time
import cv2
import numpy as np
import logging

from face_swap import FaceSwapper, QualityMode

logger = logging.getLogger("personaforge.realtime.processor")

class RealTimeProcessor:
    def __init__(self, source_image_path: str, model_name: str = "inswapper_128.onnx"):
        self.swapper = FaceSwapper(model_name=model_name, use_gpu=True)
        self.source_face = self.swapper.get_source_face(source_image_path)
        self.quality = QualityMode.FAST
        
        # Performance tracking
        self.last_process_time = 0.0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.start_time = time.time()
        
        # Adaptive thresholds
        self.LATENCY_THRESHOLD = 0.150 # 150ms
        
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        return {
            "fps": round(fps, 1),
            "latency_ms": round(self.last_process_time * 1000, 1),
            "dropped": self.frames_dropped,
            "quality": self.quality.value
        }

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        
        if self.source_face is None:
            return frame

        # Downscale for real-time processing if needed (e.g. to 480p)
        h, w = frame.shape[:2]
        TARGET_HEIGHT = 480
        scale = 1.0
        if h > TARGET_HEIGHT:
            scale = TARGET_HEIGHT / h
            small_w = int(w * scale)
            process_frame = cv2.resize(frame, (small_w, TARGET_HEIGHT))
        else:
            process_frame = frame.copy()

        # Face detection
        faces = self.swapper._app.get(process_frame)
        if not faces:
            self.last_process_time = time.perf_counter() - t0
            self._adjust_quality()
            return frame

        # Sort by largest face
        faces.sort(key=lambda f: max(0.0, float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))), reverse=True)
        target_face = faces[0]

        try:
            # Perform swap
            result = self.swapper._swap_adapter.swap_face(process_frame, target_face, self.source_face)
            
            # Upscale back to original resolution if scaled
            if scale != 1.0:
                result = cv2.resize(result, (w, h))
                
            self.frames_processed += 1
            self.last_process_time = time.perf_counter() - t0
            self._adjust_quality()
            return result
        except Exception as e:
            logger.error("RealTime Swap failed: %s", e)
            self.last_process_time = time.perf_counter() - t0
            return frame

    def _adjust_quality(self):
        """Auto-adjust quality to maintain latency under threshold."""
        if self.last_process_time > self.LATENCY_THRESHOLD and self.quality != QualityMode.FAST:
            logger.warning("Latency spike (%dms). Degrading quality to FAST.", int(self.last_process_time * 1000))
            self.quality = QualityMode.FAST
            
