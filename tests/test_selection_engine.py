import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.selection.dashboard import generate_dashboard
from backend.app.selection.engine import SmartFaceSelector, _calculate_frontal_score
from backend.app.selection.models import (
    FaceProfile,
    MediaAnalysisResponse,
    SelectionMode,
    SelectionReport,
    VideoMetadata,
)
from main import OUTPUTS_DIR, app

client = TestClient(app)


class MockFace:
    def __init__(self, bbox, embedding, det_score=0.9, kps=None):
        self.bbox = bbox
        self.embedding = embedding
        self.det_score = det_score
        self.kps = kps


class MockFaceAnalysisApp:
    def __init__(self, faces_per_call=None):
        self.faces = faces_per_call or []
        self.calls = 0

    def get(self, frame):
        self.calls += 1
        return self.faces


@pytest.fixture
def temp_out_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def selector(temp_out_dir):
    return SmartFaceSelector(face_analysis_app=None, output_dir=temp_out_dir)


@pytest.fixture
def sample_video_path(temp_out_dir):
    """Creates a tiny 1-second 30fps synthetic video for test execution."""
    vid_path = temp_out_dir / "test_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 30.0, (640, 480))
    for i in range(30):
        frame = np.full((480, 640, 3), fill_value=int(50 + (i * 2)), dtype=np.uint8)
        # Draw some contrast
        cv2.rectangle(frame, (100, 100), (300, 300), (200, 200, 200), -1)
        out.write(frame)
    out.release()
    return str(vid_path)


def test_cluster_faces(selector):
    # Two distinct identities
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])

    faces = [
        {
            "frame_idx": 0,
            "face": MockFace([0, 0, 50, 50], emb1, 0.9, None),
            "crop": np.zeros((50, 50, 3)),
            "bbox_area": 2500.0,
        },
        {
            "frame_idx": 1,
            "face": MockFace([0, 0, 50, 50], emb1, 0.9, None),
            "crop": np.zeros((50, 50, 3)),
            "bbox_area": 2500.0,
        },
        {
            "frame_idx": 0,
            "face": MockFace([0, 0, 30, 30], emb2, 0.8, None),
            "crop": np.zeros((30, 30, 3)),
            "bbox_area": 900.0,
        },
    ]

    clusters = selector.cluster_faces(faces, threshold=0.45)

    assert len(clusters) == 2
    # person_1 must be the more frequent identity
    assert "person_1" in clusters
    assert "person_2" in clusters
    assert len(clusters["person_1"]) == 2
    assert len(clusters["person_2"]) == 1


def test_calculate_frontal_score():
    # Symmetrical frontal landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    kps_frontal = np.array(
        [
            [40.0, 40.0],  # left eye
            [80.0, 40.0],  # right eye
            [60.0, 60.0],  # nose perfectly centered at x=60
            [45.0, 80.0],  # left mouth
            [75.0, 80.0],  # right mouth
        ]
    )
    face_frontal = MockFace([20, 20, 100, 100], None, det_score=0.95, kps=kps_frontal)
    score_frontal = _calculate_frontal_score(face_frontal, bbox_area=6400.0)

    # Asymmetrical turned face: nose pushed to the right side
    kps_turned = np.array(
        [
            [40.0, 40.0],
            [80.0, 40.0],
            [78.0, 60.0],  # nose close to right eye
            [50.0, 80.0],
            [80.0, 80.0],
        ]
    )
    face_turned = MockFace([20, 20, 100, 100], None, det_score=0.95, kps=kps_turned)
    score_turned = _calculate_frontal_score(face_turned, bbox_area=6400.0)

    assert score_frontal > score_turned
    assert score_frontal >= 80.0


def test_calculate_mouth_variance(selector):
    f1 = MockFace([0, 0, 100, 100], None, 0.9, np.array([[20, 20], [80, 20], [50, 50], [30, 80], [70, 80]]))
    f2 = MockFace([0, 0, 100, 100], None, 0.9, np.array([[20, 20], [80, 20], [50, 50], [30, 90], [70, 90]]))

    items = [{"face": f1}, {"face": f2}]
    var = selector._calculate_mouth_variance(items)
    assert var > 0.0


def test_calculate_profiles_and_thumbnails(selector, temp_out_dir):
    emb1 = np.array([1.0, 0.0, 0.0])
    crop_good = np.ones((60, 60, 3), dtype=np.uint8) * 200
    crop_bad = np.ones((40, 40, 3), dtype=np.uint8) * 100

    items = [
        {
            "frame_idx": 0,
            "face": MockFace([10, 10, 70, 70], emb1, 0.95),
            "crop": crop_good,
            "bbox_area": 3600.0,
            "frontal_score": 92.0,
        },
        {
            "frame_idx": 1,
            "face": MockFace([10, 10, 50, 50], emb1, 0.80),
            "crop": crop_bad,
            "bbox_area": 1600.0,
            "frontal_score": 65.0,
        },
    ]

    clusters = {"person_1": items}
    profiles = selector.calculate_profiles(clusters, total_sampled_frames=2, prefix="sess123")

    assert len(profiles) == 1
    p = profiles[0]
    assert p.face_id == "person_1"
    assert p.person_label == "Detected Person 1"
    assert p.sample_count == 2
    assert p.visibility_duration == 100.0
    assert p.thumbnail_url == "/selection/thumbnails/sess123_person_1_thumb.jpg"
    assert p.representative_embedding is not None

    # Verify thumbnail file was written to disk
    thumb_path = temp_out_dir / "sess123_person_1_thumb.jpg"
    assert thumb_path.exists()
    assert thumb_path.stat().st_size > 0


def test_rank_and_select(selector):
    p1 = FaceProfile(
        face_id="person_1",
        person_label="Detected Person 1",
        average_area=1000.0,
        visibility_duration=50.0,
        detection_confidence=99.0,
        speaking_score=10.0,
        frontal_score=90.0,
    )
    p2 = FaceProfile(
        face_id="person_2",
        person_label="Detected Person 2",
        average_area=500.0,
        visibility_duration=100.0,
        detection_confidence=95.0,
        speaking_score=50.0,
        frontal_score=75.0,
    )

    profiles = [p1, p2]

    best, _conf = selector.rank_and_select(profiles, SelectionMode.LARGEST)
    assert best.face_id == "person_1"

    best, _conf = selector.rank_and_select(profiles, SelectionMode.MOST_VISIBLE)
    assert best.face_id == "person_2"

    best, _conf = selector.rank_and_select(profiles, SelectionMode.MAIN_SPEAKER)
    assert best.face_id == "person_2"

    best, _conf = selector.rank_and_select(profiles, SelectionMode.HIGHEST_CONFIDENCE)
    assert best.face_id == "person_1"


def test_evaluate_warnings(selector):
    # Low resolution, dark frame, no faces
    meta_low = VideoMetadata(
        duration=10.0,
        fps=30.0,
        total_frames=300,
        width=320,
        height=240,
        orientation="landscape",
        aspect_ratio=1.3333,
        codec="h264",
        file_size_mb=1.2,
    )
    frame_stats = {"avg_luminance": 25.0, "avg_sharpness": 40.0, "sampled_frames_count": 10}
    warnings, _rec_mode = selector.evaluate_warnings(meta_low, frame_stats, [])

    assert any("Low video resolution" in w for w in warnings)
    assert any("Low scene illumination" in w for w in warnings)
    assert any("Significant motion blur" in w for w in warnings)
    assert any("No stable human faces detected" in w for w in warnings)


def test_analyze_media_end_to_end(temp_out_dir, sample_video_path):
    emb = np.array([0.5, 0.5, 0.5])
    kps = np.array([[10, 10], [50, 10], [30, 30], [20, 50], [40, 50]])
    mock_face = MockFace([50, 50, 150, 150], emb, det_score=0.98, kps=kps)
    mock_app = MockFaceAnalysisApp(faces_per_call=[mock_face])

    selector = SmartFaceSelector(face_analysis_app=mock_app, output_dir=temp_out_dir)
    res = selector.analyze_media(
        job_id="test_job_1",
        video_path=sample_video_path,
        mode=SelectionMode.HIGHEST_CONFIDENCE,
        session_id="test_sess",
    )

    assert isinstance(res, MediaAnalysisResponse)
    assert res.job_id == "test_job_1"
    assert res.session_id == "test_sess"
    assert res.video_metadata is not None
    assert res.video_metadata.width == 640
    assert res.video_metadata.height == 480
    assert len(res.detected_identities) == 1
    assert res.selected_face_id == "person_1"
    assert res.selected_person_label == "Detected Person 1"
    assert res.detected_identities[0].thumbnail_url is not None


def test_generate_dashboard(temp_out_dir):
    p1 = FaceProfile(
        face_id="person_1",
        person_label="Detected Person 1",
        average_area=1000.0,
        visibility_duration=50.0,
        detection_confidence=99.0,
        speaking_score=10.0,
    )

    report = SelectionReport(
        job_id="test_job",
        selected_face_id="person_1",
        selected_person_label="Detected Person 1",
        selection_mode=SelectionMode.LARGEST,
        confidence_score=100.0,
        profiles=[p1],
    )

    path = generate_dashboard(report, temp_out_dir, "test")
    assert path.exists()
    assert path.name == "selection_dashboard_test.html"


def test_api_media_analyze_validation():
    # Calling POST /media/analyze without video or session_id must return 400
    res = client.post("/media/analyze")
    assert res.status_code == 400
    assert "Either 'video' file or 'session_id'" in res.json().get("detail", "")


def test_api_thumbnails_endpoint():
    # Test 404 for missing thumbnail
    res_404 = client.get("/selection/thumbnails/nonexistent_thumb_123.jpg")
    assert res_404.status_code == 404

    # Create dummy thumbnail in thumbnails dir
    thumb_dir = OUTPUTS_DIR / "selection_thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    test_thumb = thumb_dir / "api_test_thumb.jpg"
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(test_thumb), dummy_img)

    try:
        # Test fetching with .jpg
        res = client.get("/selection/thumbnails/api_test_thumb.jpg")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"

        # Test fetching without extension
        res_no_ext = client.get("/selection/thumbnails/api_test_thumb")
        assert res_no_ext.status_code == 200
        assert res_no_ext.headers["content-type"] == "image/jpeg"
    finally:
        test_thumb.unlink(missing_ok=True)
