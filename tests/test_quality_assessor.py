import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.quality.assessor import FaceQualityAssessor
from backend.app.quality.blur_detection import LaplacianBlurDetector
from backend.app.quality.confidence import FaceConfidenceEvaluator
from backend.app.quality.dashboard import generate_dashboard
from backend.app.quality.sharpness import SobelSharpnessEvaluator
from backend.app.quality.stability import LandmarkStabilityEvaluator


# Create a mock Face object to simulate InsightFace output
class MockFace:
    def __init__(self, bbox, pose, det_score, kps=None):
        self.bbox = bbox
        self.pose = pose
        self.det_score = det_score
        self.kps = (
            kps
            if kps is not None
            else np.array(
                [
                    [150.0, 150.0],  # right eye
                    [250.0, 150.0],  # left eye
                    [200.0, 200.0],  # nose
                    [160.0, 260.0],  # right mouth corner
                    [240.0, 260.0],  # left mouth corner
                ],
                dtype=np.float32,
            )
        )


class MockFaceAnalysis:
    def __init__(self):
        pass

    def get(self, image):
        # We simulate finding one face in the image
        return [
            MockFace(
                bbox=[100, 100, 300, 300],  # x1, y1, x2, y2
                pose=[0.0, 0.0, 0.0],  # pitch, yaw, roll
                det_score=0.99,
            )
        ]


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
    cv2.line(clear_img, (150, 150), (250, 250), (0, 0, 0), 5)  # add edges for sharpness/blur

    # 2. Blurry Image
    blurry_img = cv2.GaussianBlur(clear_img, (25, 25), 0)

    # 3. Dark Image
    dark_img = clear_img.copy() // 4

    cv2.imwrite(os.path.join(temp_dir, "clear.jpg"), clear_img)
    cv2.imwrite(os.path.join(temp_dir, "blurry.jpg"), blurry_img)
    cv2.imwrite(os.path.join(temp_dir, "dark.jpg"), dark_img)

    yield temp_dir

    shutil.rmtree(temp_dir)


# ─── Legacy & Core FaceQualityAssessor Tests ───────────────────────────────────


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

    cb, _cc = assessor.calculate_lighting(clear_img)
    db, _dc = assessor.calculate_lighting(dark_img)

    assert db < cb
    assert db < 40.0


def test_assess_image_overall(assessor, benchmark_dir):
    clear_path = os.path.join(benchmark_dir, "clear.jpg")
    report = assessor.assess_image(clear_path)

    assert report.quality_score > 0
    assert report.metrics.face_angle == 100.0  # based on our mock pose
    assert report.metrics.occlusion == 99.0  # based on our mock det_score
    assert len(report.recommendations) >= 0


def test_dashboard_generation(assessor, benchmark_dir):
    clear_path = os.path.join(benchmark_dir, "clear.jpg")
    report = assessor.assess_image(clear_path)

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td)
        chart_path = generate_dashboard(report, out_path, "test_dash")
        assert chart_path.exists()
        assert chart_path.name == "quality_dashboard_test_dash.html"
        assert "plotly" in chart_path.read_text(encoding="utf-8").lower()


# ─── Modular Component Tests ───────────────────────────────────────────────────


def test_sobel_sharpness_evaluator(benchmark_dir):
    clear_img = cv2.imread(os.path.join(benchmark_dir, "clear.jpg"))
    blurry_img = cv2.imread(os.path.join(benchmark_dir, "blurry.jpg"))

    clear_sharp = SobelSharpnessEvaluator.calculate_sharpness(clear_img)
    blurry_sharp = SobelSharpnessEvaluator.calculate_sharpness(blurry_img)

    assert clear_sharp > blurry_sharp
    assert SobelSharpnessEvaluator.calculate_sharpness(None) == 0.0
    assert SobelSharpnessEvaluator.calculate_sharpness(np.empty((0, 0))) == 0.0


def test_laplacian_blur_detector(benchmark_dir):
    clear_img = cv2.imread(os.path.join(benchmark_dir, "clear.jpg"))
    blurry_img = cv2.imread(os.path.join(benchmark_dir, "blurry.jpg"))

    raw_var_clear = LaplacianBlurDetector.compute_variance(clear_img)
    raw_var_blurry = LaplacianBlurDetector.compute_variance(blurry_img)
    assert raw_var_clear > raw_var_blurry

    # Calibrated logarithmic sharpness per docs/metrics.md
    norm_sharp_clear = LaplacianBlurDetector.calculate_calibrated_sharpness(clear_img)
    norm_sharp_blurry = LaplacianBlurDetector.calculate_calibrated_sharpness(blurry_img)
    assert norm_sharp_clear > norm_sharp_blurry
    assert 0.0 <= norm_sharp_clear <= 100.0

    # Edge cases
    assert LaplacianBlurDetector.compute_variance(None) == 0.0
    assert LaplacianBlurDetector.calculate_blur_score(None) == 0.0
    assert LaplacianBlurDetector.calculate_calibrated_sharpness(None) == 0.0


def test_landmark_stability_evaluator():
    # Base 5-point landmark (IOD = 100.0 pixels between eye indices 0 and 1)
    kps1 = np.array(
        [
            [150.0, 150.0],
            [250.0, 150.0],
            [200.0, 200.0],
            [160.0, 260.0],
            [240.0, 260.0],
        ],
        dtype=np.float32,
    )

    iod = LandmarkStabilityEvaluator.compute_iod(kps1)
    assert np.isclose(iod, 100.0)

    # Smooth movement: 2 pixel shift in x -> 2 / 100 = 0.02 IOD (< 0.05 smooth threshold)
    kps2 = kps1.copy()
    kps2[:, 0] += 2.0
    jitter_smooth = LandmarkStabilityEvaluator.compute_frame_jitter(kps1, kps2)
    assert np.isclose(jitter_smooth, 0.02, atol=1e-3)

    # Sequence with smooth movement
    seq_smooth = [kps1, kps2, kps1 + 3.0, kps1 + 1.0]
    metrics_smooth = LandmarkStabilityEvaluator.evaluate_sequence(seq_smooth)
    assert metrics_smooth.mean_jitter_iod < 0.05
    assert metrics_smooth.jitter_spikes_count == 0
    assert metrics_smooth.stability_score >= 80.0

    # Sequence with jitter spike (shift by 25 pixels -> 0.25 IOD > 0.18 spike threshold)
    kps_spike = kps1 + 25.0
    seq_spiky = [kps1, kps_spike, kps1]
    metrics_spiky = LandmarkStabilityEvaluator.evaluate_sequence(seq_spiky)
    assert metrics_spiky.jitter_spikes_count > 0
    assert metrics_spiky.stability_score < metrics_smooth.stability_score

    # Edge cases
    assert LandmarkStabilityEvaluator.evaluate_sequence([]).stability_score == 100.0
    assert LandmarkStabilityEvaluator.evaluate_sequence([kps1]).stability_score == 100.0


def test_face_confidence_evaluator():
    # Lighting
    white_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    b, _c = FaceConfidenceEvaluator.calculate_lighting(white_img)
    assert np.isclose(b, 100.0)  # exactly at ideal 128

    # Face size score
    # 200x200 bbox in 500x500 image -> 40,000 / 250,000 = 16% (ideal range 10-30%)
    size_score = FaceConfidenceEvaluator.calculate_face_size_score([100, 100, 300, 300], (500, 500, 3))
    assert size_score == 100.0

    # Tiny bbox: 10x10 in 1000x1000 -> 0.01% -> scaled down
    tiny_score = FaceConfidenceEvaluator.calculate_face_size_score([0, 0, 10, 10], (1000, 1000, 3))
    assert tiny_score < 10.0

    # Face pose angle evaluation (Safeguard 7)
    mock_frontal = MockFace(bbox=[0, 0, 10, 10], pose=[0.0, 0.0, 0.0], det_score=0.95)
    angle_frontal = FaceConfidenceEvaluator.calculate_face_angle_score(mock_frontal)
    assert angle_frontal == 100.0

    # Without pose attribute, returns nominal 80.0 without speculative penalty
    class MockNoPose:
        def __init__(self):
            self.bbox = [0, 0, 10, 10]
            self.det_score = 0.90

    assert FaceConfidenceEvaluator.calculate_face_angle_score(MockNoPose()) == 80.0


def test_assess_video_sequence(assessor, benchmark_dir):
    clear_img = cv2.imread(os.path.join(benchmark_dir, "clear.jpg"))
    frames = [clear_img, clear_img, clear_img]

    vid_report = assessor.assess_video_sequence(frames)
    assert vid_report.total_frames_analyzed == 3
    assert vid_report.overall_quality_score > 0
    assert vid_report.temporal_stability.stability_score == 100.0
    assert len(vid_report.recommendations) >= 1

    # Empty sequence
    empty_report = assessor.assess_video_sequence([])
    assert empty_report.total_frames_analyzed == 0
    assert empty_report.overall_quality_score == 0.0
