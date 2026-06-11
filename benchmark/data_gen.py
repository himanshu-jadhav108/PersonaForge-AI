import cv2
import numpy as np
from pathlib import Path
import os

def generate_dummy_video(output_path: Path, resolution: str, face_count: int, frames: int = 30):
    """
    Generates a dummy video with moving rectangles to simulate faces for benchmarking overheads.
    """
    res_map = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160)
    }
    
    width, height = res_map.get(resolution.lower(), (1280, 720))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
    
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw moving "faces"
        for f_idx in range(face_count):
            x = 100 + (f_idx * 200) + (i * 5)
            y = 100 + (f_idx * 150)
            
            # keep within bounds
            x = x % (width - 150)
            y = y % (height - 150)
            
            # Draw a face-like rectangle
            cv2.rectangle(frame, (x, y), (x+150, y+150), (200, 200, 200), -1)
            # Add some features
            cv2.circle(frame, (x+40, y+40), 10, (0, 0, 0), -1) # Eye
            cv2.circle(frame, (x+110, y+40), 10, (0, 0, 0), -1) # Eye
            cv2.rectangle(frame, (x+50, y+100), (x+100, y+120), (0, 0, 0), -1) # Mouth
            
        out.write(frame)
        
    out.release()
    return output_path

def generate_dummy_source_face(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 220, 200), -1)
    cv2.circle(frame, (200, 200), 20, (0, 0, 0), -1) 
    cv2.circle(frame, (300, 200), 20, (0, 0, 0), -1) 
    cv2.rectangle(frame, (200, 300), (300, 320), (0, 0, 0), -1)
    cv2.imwrite(str(output_path), frame)
    return output_path
