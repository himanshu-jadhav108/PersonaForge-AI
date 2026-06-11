import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
import shutil

from backend.app.quality.assessor import FaceQualityAssessor
from backend.app.quality.dashboard import generate_dashboard

# Create a mock Face object to simulate InsightFace output
class MockFace:
    def __init__(self, bbox, pose, det_score):
        self.bbox = bbox
        self.pose = pose
        self.det_score = det_score

class MockFaceAnalysis:
    def __init__(self):
        pass
    def get(self, image):
        # We simulate finding one face in the image
        return [MockFace(
            bbox=[100, 100, 300, 300], # x1, y1, x2, y2
            pose=[0.0, 0.0, 0.0],      # pitch, yaw, roll
            det_score=0.99
        )]

@pytest.fixture
def assessor():
    app = MockFaceAnalysis()
    return FaceQualityAssessor(face_analysis_app=app)

@pytest.fixture
def benchmark_dir():
    # Setup temporary benchmark dir
    temp_dir = tempfile.mkdtemp()
    
    # 1. Clear Image
    clear_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(clear_img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.line(clear_img, (150, 150), (250, 250), (0, 0, 0), 5) # add edges for sharpness/blur
    
    # 2. Blurry Image
    blurry_img = cv2.GaussianBlur(clear_img, (25, 25), 0)
    
    # 3. Dark Image
    dark_img = clear_img.copy() // 4
    
    cv2.imwrite(os.path.join(temp_dir, "clear.jpg"), clear_img)
    cv2.imwrite(os.path.join(temp_dir, "blurry.jpg"), blurry_img)
    cv2.imwrite(os.path.join(temp_dir, "dark.jpg"), dark_img)
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)


def test_blur_score(assessor, benchmark_dir):
    clear_img = cv2.imread(os.path.join(benchmark_dir, "clear.jpg"))
    blurry_img = cv2.imread(os.path.join(benchmark_dir, "blurry.jpg"))
    
    clear_score = assessor.calculate_blur_score(clear_img)
    blurry_score = assessor.calculate_blur_score(blurry_img)
    
    assert clear_score > blurry_score
    assert blurry_score < 50.0

def test_lighting_score(assessor, benchmark_dir):
    clear_img = cv2.imread(os.path.join(benchmark_dir, "clear.jpg"))
    dark_img = cv2.imread(os.path.join(benchmark_dir, "dark.jpg"))
    
    cb, cc = assessor.calculate_lighting(clear_img)
    db, dc = assessor.calculate_lighting(dark_img)
    
    assert db < cb
    assert db < 40.0

def test_assess_image_overall(assessor, benchmark_dir):
    clear_path = os.path.join(benchmark_dir, "clear.jpg")
    report = assessor.assess_image(clear_path)
    
    assert report.quality_score > 0
    assert report.metrics.face_angle == 100.0  # based on our mock pose
    assert report.metrics.occlusion == 99.0    # based on our mock det_score
    assert len(report.recommendations) >= 0

def test_dashboard_generation(assessor, benchmark_dir):
    clear_path = os.path.join(benchmark_dir, "clear.jpg")
    report = assessor.assess_image(clear_path)
    
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td)
        chart_path = generate_dashboard(report, out_path, "test_dash")
        assert chart_path.exists()
        assert chart_path.name == "quality_dashboard_test_dash.html"
