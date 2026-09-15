"""
tests/test_multiperson_swap.py — Unit and integration tests for Phase 11 (Multi-Person Simultaneous Face Swapping).
"""

import io
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import UPLOADS_DIR, app
from utils.database import JobDB

client = TestClient(app)


def test_multi_target_matching_logic():
    """Verify greedy cosine distance matching pairs each detected face with the right source face."""
    # 512D unit embeddings
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0

    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0

    src1 = MagicMock(name="source_face_1")
    src2 = MagicMock(name="source_face_2")

    target_mapping = [
        (emb1, src1),
        (emb2, src2),
    ]

    # Create 3 detected faces: df1 matches emb1, df2 matches emb2, df3 is orthogonal distractor
    df1 = MagicMock(name="detected_1")
    df1_emb = np.zeros(512, dtype=np.float32)
    df1_emb[0] = 0.95
    df1_emb[2] = 0.05
    df1.embedding = df1_emb / np.linalg.norm(df1_emb)
    df1.bbox = [10, 10, 50, 50]

    df2 = MagicMock(name="detected_2")
    df2_emb = np.zeros(512, dtype=np.float32)
    df2_emb[1] = 0.92
    df2_emb[3] = 0.08
    df2.embedding = df2_emb / np.linalg.norm(df2_emb)
    df2.bbox = [100, 100, 150, 150]

    df3 = MagicMock(name="detected_3")
    df3_emb = np.zeros(512, dtype=np.float32)
    df3_emb[10] = 1.0  # orthogonal
    df3.embedding = df3_emb
    df3.bbox = [200, 200, 250, 250]

    all_faces = [df1, df2, df3]

    matched_pairs = []
    matched_targets = set()
    for df in all_faces:
        df_emb_arr = np.array(df.embedding, dtype=np.float32).flatten()
        df_norm = float(np.linalg.norm(df_emb_arr))
        best_idx = None
        best_sim = -1.0
        for idx, (t_emb, _t_src) in enumerate(target_mapping):
            if idx in matched_targets:
                continue
            t_norm = float(np.linalg.norm(t_emb))
            if t_norm > 0:
                sim = float(np.dot(df_emb_arr, t_emb) / (df_norm * t_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

        if best_idx is not None and best_sim >= 0.40:
            matched_targets.add(best_idx)
            matched_pairs.append((df, target_mapping[best_idx][1]))

    assert len(matched_pairs) == 2
    assert matched_pairs[0][0] == df1
    assert matched_pairs[0][1] == src1
    assert matched_pairs[1][0] == df2
    assert matched_pairs[1][1] == src2


def test_upload_auxiliary_source_endpoint(tmp_path: Path):
    """Verify uploading an auxiliary source face image for a specific target person."""
    session_id = uuid.uuid4().hex
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create a valid PNG byte stream
        png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 30
        response = client.post(
            f"/upload/source?session_id={session_id}&target_face_id=person_2",
            files={"image": ("source_p2.png", io.BytesIO(png_header), "image/png")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["target_face_id"] == "person_2"
        assert (session_dir / data["source_filename"]).exists()

    finally:
        import shutil

        shutil.rmtree(session_dir, ignore_errors=True)


def test_process_multi_endpoint_and_routing():
    """Verify /process/multi triggers multi-person background job."""
    session_id = uuid.uuid4().hex
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create dummy session source and target
        (session_dir / "source_face.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01" + b"\x00" * 30)
        (session_dir / "target_video.mp4").write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 30)
        (session_dir / "identities.json").write_text(
            json.dumps(
                {
                    "person_1": {"person_label": "Person 1", "embedding": [0.1] * 512},
                    "person_2": {"person_label": "Person 2", "embedding": [0.2] * 512},
                }
            ),
            encoding="utf-8",
        )

        target_mappings = json.dumps({"person_1": "source_face.jpg", "person_2": "source_face.jpg"})

        with patch("main.BackgroundTasks.add_task") as mock_add_task:
            response = client.post(
                f"/process/multi?session_id={session_id}&target_mappings={target_mappings}&quality=fast"
            )

            assert response.status_code == 200
            res = response.json()
            assert "job_id" in res
            assert "Multi-person simultaneous processing started" in res["message"]
            assert mock_add_task.called

    finally:
        import shutil

        shutil.rmtree(session_dir, ignore_errors=True)
