import pytest
import numpy as np
from pathlib import Path
import tempfile

from backend.app.selection.models import SelectionMode, FaceProfile
from backend.app.selection.engine import SmartFaceSelector
from backend.app.selection.dashboard import generate_dashboard

class MockFace:
    def __init__(self, bbox, embedding, det_score, kps):
        self.bbox = bbox
        self.embedding = embedding
        self.det_score = det_score
        self.kps = kps

@pytest.fixture
def temp_out_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

@pytest.fixture
def selector(temp_out_dir):
    return SmartFaceSelector(face_analysis_app=None, output_dir=temp_out_dir)

def test_cluster_faces(selector):
    # Two identities
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    
    faces = [
        {'frame_idx': 0, 'face': MockFace([0,0,10,10], emb1, 0.9, None), 'crop': np.zeros((10,10,3))},
        {'frame_idx': 1, 'face': MockFace([0,0,10,10], emb1, 0.9, None), 'crop': np.zeros((10,10,3))},
        {'frame_idx': 0, 'face': MockFace([0,0,10,10], emb2, 0.8, None), 'crop': np.zeros((10,10,3))},
    ]
    
    clusters = selector.cluster_faces(faces, threshold=0.5)
    
    assert len(clusters) == 2
    # one cluster should have 2 items, the other 1
    lens = sorted([len(v) for v in clusters.values()])
    assert lens == [1, 2]

def test_calculate_mouth_variance(selector):
    # Create mock face with kps (5 points)
    # kps = [leye, reye, nose, lmouth, rmouth]
    
    f1 = MockFace([0,0,100,100], None, 0.9, np.array([[20,20], [80,20], [50,50], [30,80], [70,80]]))
    f2 = MockFace([0,0,100,100], None, 0.9, np.array([[20,20], [80,20], [50,50], [30,90], [70,90]])) # mouth moved
    
    items = [{'face': f1}, {'face': f2}]
    
    var = selector._calculate_mouth_variance(items)
    assert var > 0.0

def test_rank_and_select(selector):
    p1 = FaceProfile(
        face_id="id1",
        average_area=1000.0,
        visibility_duration=50.0,
        detection_confidence=99.0,
        speaking_score=10.0
    )
    p2 = FaceProfile(
        face_id="id2",
        average_area=500.0,
        visibility_duration=100.0, # appears more
        detection_confidence=95.0,
        speaking_score=50.0 # main speaker
    )
    
    profiles = [p1, p2]
    
    best, conf = selector.rank_and_select(profiles, SelectionMode.LARGEST)
    assert best.face_id == "id1"
    
    best, conf = selector.rank_and_select(profiles, SelectionMode.MOST_VISIBLE)
    assert best.face_id == "id2"
    
    best, conf = selector.rank_and_select(profiles, SelectionMode.MAIN_SPEAKER)
    assert best.face_id == "id2"
    
    best, conf = selector.rank_and_select(profiles, SelectionMode.HIGHEST_CONFIDENCE)
    assert best.face_id == "id1"

def test_generate_dashboard(temp_out_dir):
    from backend.app.selection.models import SelectionReport
    
    p1 = FaceProfile(
        face_id="id1",
        average_area=1000.0,
        visibility_duration=50.0,
        detection_confidence=99.0,
        speaking_score=10.0
    )
    
    report = SelectionReport(
        job_id="test_job",
        selected_face_id="id1",
        selection_mode=SelectionMode.LARGEST,
        confidence_score=100.0,
        profiles=[p1]
    )
    
    path = generate_dashboard(report, temp_out_dir, "test")
    assert path.exists()
    assert path.name == "selection_dashboard_test.html"
