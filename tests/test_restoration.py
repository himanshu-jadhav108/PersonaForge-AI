from unittest.mock import MagicMock

import cv2
import numpy as np

from backend.app.models.restoration.classic_enhancer import ClassicEnhancer
from backend.app.models.restoration.codeformer_adapter import CodeFormerRestorer
from backend.app.models.restoration.factory import RestorationFactory
from backend.app.models.restoration.gfpgan_adapter import GFPGANRestorer
from backend.app.models.restoration.identity_guard import RestorationIdentityGuard

# ─── ClassicEnhancer Tests ────────────────────────────────────────────────────


def test_classic_enhancer_processing():
    enhancer = ClassicEnhancer()
    crop = np.full((128, 128, 3), 120, dtype=np.uint8)
    cv2.circle(crop, (64, 64), 30, (200, 200, 200), -1)

    enhanced = enhancer.restore_crop(crop, blend_weight=1.0)
    assert enhanced.shape == crop.shape
    assert enhancer.is_available() is True

    # Transparency check per Safeguard 2
    info = enhancer.get_model_info()
    assert info["is_ai"] is False
    assert "Bilateral" in info["description"]
    assert "Apache 2.0" in info["license"]

    # Blend weight 0.0 should return identical pixels
    zero_blend = enhancer.restore_crop(crop, blend_weight=0.0)
    assert np.array_equal(zero_blend, crop)

    # Edge cases
    assert enhancer.restore_crop(None) is None
    assert enhancer.restore_crop(np.empty((0, 0))).size == 0


# ─── GFPGAN & CodeFormer Adapter Tests ─────────────────────────────────────────


def test_gfpgan_adapter_behavior_without_weights():
    restorer = GFPGANRestorer(model_path="non_existent_models/gfpgan.onnx")
    assert restorer.is_available() is False

    info = restorer.get_model_info()
    assert info["name"] == "GFPGAN v1.4"
    assert info["is_ai"] is True
    assert "Apache 2.0" in info["license"]
    assert info["is_available"] is False

    # Graceful fallback when weights are missing
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    output = restorer.restore_crop(crop, blend_weight=1.0)
    assert np.array_equal(output, crop)


def test_codeformer_adapter_behavior_without_weights():
    restorer = CodeFormerRestorer(model_path="non_existent_models/codeformer.onnx")
    assert restorer.is_available() is False

    info = restorer.get_model_info()
    assert info["name"] == "CodeFormer"
    assert info["is_ai"] is True
    assert "Non-Commercial Research" in info["license"]
    assert info["is_available"] is False

    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    output = restorer.restore_crop(crop, blend_weight=1.0)
    assert np.array_equal(output, crop)


# ─── RestorationFactory & Safeguard 2 Transparency Tests ───────────────────────


def test_restoration_factory_safeguard2_fallback():
    # Attempting to load missing GFPGAN model with fallback=True
    restorer, status = RestorationFactory.create_restorer(
        name="gfpgan",
        fallback_to_classic=True,
        model_path="non_existent_gfpgan.onnx",
    )

    # Must fall back to ClassicEnhancer
    assert isinstance(restorer, ClassicEnhancer)
    # Strictly transparent reporting
    assert status["ai_restoration"] == "Unavailable"
    assert status["classic_enhancement"] == "Enabled (Bilateral / Unsharp)"
    assert status["is_ai"] is False
    assert status["active_restorer"] == "ClassicEnhancer"
    assert "AI Restoration: Unavailable" in status["status_message"]


def test_restoration_factory_explicit_classic():
    restorer, status = RestorationFactory.create_restorer(name="classic")
    assert isinstance(restorer, ClassicEnhancer)
    assert status["is_ai"] is False
    assert status["ai_restoration"] == "Unavailable"


def test_restoration_factory_disabled():
    _restorer, status = RestorationFactory.create_restorer(name="none")
    assert status["ai_restoration"] == "Unavailable"
    assert status["classic_enhancement"] == "Disabled"
    assert status["active_restorer"] == "None"


# ─── RestorationIdentityGuard Tests ────────────────────────────────────────────


def test_restoration_identity_guard_safe():
    guard = RestorationIdentityGuard(max_allowed_drop=0.08)

    source_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pre_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    restored_crop = np.ones((128, 128, 3), dtype=np.uint8)

    # Mock FaceAnalysis app returning identical embeddings for pre and restored
    mock_app = MagicMock()
    mock_face = MagicMock()
    mock_face.embedding = np.array([0.98, 0.1, 0.0], dtype=np.float32)
    mock_face.bbox = [10, 10, 100, 100]
    mock_app.get.return_value = [mock_face]

    guarded_crop, drift_detected, safe_w, drop = guard.guard_restoration(
        source_embedding=source_emb,
        pre_restored_crop=pre_crop,
        restored_crop=restored_crop,
        face_analysis_app=mock_app,
    )

    assert drift_detected is False
    assert safe_w == 1.0
    assert drop == 0.0
    assert np.array_equal(guarded_crop, restored_crop)


def test_restoration_identity_guard_drift_detected():
    guard = RestorationIdentityGuard(max_allowed_drop=0.08)

    source_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pre_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    restored_crop = np.full((128, 128, 3), 255, dtype=np.uint8)

    # Mock app returning high similarity for pre_crop, but degraded similarity for restored_crop
    mock_app = MagicMock()
    face_pre = MagicMock()
    face_pre.embedding = np.array([0.95, 0.0, 0.0], dtype=np.float32)  # sim = 0.95
    face_pre.bbox = [10, 10, 100, 100]

    face_post = MagicMock()
    face_post.embedding = np.array([0.70, 0.71, 0.0], dtype=np.float32)  # sim = 0.70 (drop = 0.25 >= 0.08)
    face_post.bbox = [10, 10, 100, 100]

    mock_app.get.side_effect = [[face_pre], [face_post]]

    guarded_crop, drift_detected, safe_w, drop = guard.guard_restoration(
        source_embedding=source_emb,
        pre_restored_crop=pre_crop,
        restored_crop=restored_crop,
        face_analysis_app=mock_app,
    )

    assert drift_detected is True
    assert safe_w < 1.0
    assert drop > 0.08
    # The output is re-blended, so it should not be pure 255
    assert not np.array_equal(guarded_crop, restored_crop)
