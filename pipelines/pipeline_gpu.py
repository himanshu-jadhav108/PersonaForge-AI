"""
pipeline_gpu.py — PersonaForge AI · GPU Pipeline Wrapper

Thin symmetry module: delegates directly to FaceSwapper.process_video()
which is the original, unmodified GPU pipeline in face_swap.py.

Nothing in this file alters GPU behaviour.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from face_swap import FaceSwapper, QualityMode


def process_video_gpu(
    swapper: "FaceSwapper",
    source_face,
    frames_dir:  str,
    output_dir:  str,
    quality:     "QualityMode",
    face_index:  int         = -1,
    max_frames:  Optional[int] = None,
    progress_start: int      = 40,
    progress_end:   int      = 80,
    db_manager               = None,
    job_id:      str         = None,
    identity_validator       = None,
) -> tuple[int, int]:
    """
    Delegate to the original GPU processing loop unchanged.
    All parameters pass through without modification.
    """
    return swapper.process_video(
        source_face     = source_face,
        frames_dir      = frames_dir,
        output_dir      = output_dir,
        quality         = quality,
        face_index      = face_index,
        max_frames      = max_frames,
        progress_start  = progress_start,
        progress_end    = progress_end,
        db_manager      = db_manager,
        job_id          = job_id,
        identity_validator = identity_validator,
    )
