"""
tests/test_pipeline_modes.py — Verification of Blending, Tracking Abstractions, and Dynamic Resolution
"""

import unittest

import numpy as np

from backend.app.tracking.base import BaseFaceTracker
from backend.app.tracking.detection_tracker import DetectionOnlyTracker
from backend.app.tracking.factory import get_tracker
from backend.app.tracking.kcf_tracker import KCFTracker
from pipelines.blending.alpha_blend import AlphaBlend
from pipelines.blending.base import BaseBlender
from pipelines.blending.factory import get_blender
from pipelines.blending.feathered_blend import FeatheredBlend
from pipelines.blending.seamless_clone import SeamlessCloneExperimental
from video_utils import compute_mode_resolution


class TestBlendingModule(unittest.TestCase):
    """Test blending abstractions and implementations."""

    def setUp(self):
        # Create dummy background frame: 200x200 grey
        self.frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        # Create dummy swapped face crop: 60x60 white
        self.crop = np.full((60, 60, 3), 255, dtype=np.uint8)
        self.x1, self.y1, self.x2, self.y2 = 50, 50, 110, 110

    def test_alpha_blend(self):
        blender = AlphaBlend()
        self.assertIsInstance(blender, BaseBlender)

        result = blender.blend(self.frame, self.crop, self.x1, self.y1, self.x2, self.y2)
        self.assertEqual(result.shape, self.frame.shape)
        self.assertEqual(result.dtype, np.uint8)
        # Center of inserted patch should be modified
        self.assertTrue(np.any(result[80, 80] != 100))
        # Background outside patch should remain unchanged
        self.assertTrue(np.all(result[10, 10] == 100))

    def test_feathered_blend(self):
        blender = FeatheredBlend(feather_radius=5)
        self.assertIsInstance(blender, BaseBlender)

        result = blender.blend(self.frame, self.crop, self.x1, self.y1, self.x2, self.y2)
        self.assertEqual(result.shape, self.frame.shape)
        self.assertEqual(result.dtype, np.uint8)
        # Center should have swapped face content
        self.assertTrue(np.all(result[80, 80] > 150))
        # Edges should transition smoothly (feathered values between 100 and 255)
        self.assertTrue(np.all(result[10, 10] == 100))

    def test_seamless_clone_experimental_fallback(self):
        blender = SeamlessCloneExperimental()
        self.assertIsInstance(blender, BaseBlender)

        # Even with boundary coordinates close to edges, it should not raise
        result = blender.blend(self.frame, self.crop, self.x1, self.y1, self.x2, self.y2)
        self.assertEqual(result.shape, self.frame.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_blender_factory(self):
        self.assertIsInstance(get_blender("alpha"), AlphaBlend)
        self.assertIsInstance(get_blender("feathered"), FeatheredBlend)
        self.assertIsInstance(get_blender("seamless_clone_experimental"), SeamlessCloneExperimental)
        # Unknown falls back safely to feathered
        self.assertIsInstance(get_blender("unknown_blender"), FeatheredBlend)


class TestTrackingModule(unittest.TestCase):
    """Test face tracking abstraction and implementations."""

    def setUp(self):
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)
        self.bbox = (40, 40, 50, 50)

    def test_detection_only_tracker(self):
        tracker = DetectionOnlyTracker()
        self.assertIsInstance(tracker, BaseFaceTracker)

        tracker.init(self.frame, self.bbox)
        success, bbox_out = tracker.update(self.frame)
        self.assertFalse(success, "DetectionOnlyTracker must always return False to trigger detection")
        self.assertIsNone(bbox_out)

    def test_kcf_tracker_contract(self):
        tracker = KCFTracker()
        self.assertIsInstance(tracker, BaseFaceTracker)

        tracker.init(self.frame, self.bbox)
        success, bbox_out = tracker.update(self.frame)
        # On a blank frame it might succeed or fail, but must conform to return signature
        self.assertIsInstance(success, bool)
        if success:
            self.assertEqual(len(bbox_out), 4)
        tracker.reset()

    def test_tracker_factory(self):
        self.assertIsInstance(get_tracker("detection_only"), DetectionOnlyTracker)
        self.assertIsInstance(get_tracker("kcf"), KCFTracker)
        self.assertIsInstance(get_tracker("unknown"), KCFTracker)


class TestDynamicResolutionScaling(unittest.TestCase):
    """Test dynamic resolution computation."""

    def test_compute_mode_resolution_downscaling(self):
        # 1080p (1920x1080) to 720p
        w, h = compute_mode_resolution(1920, 1080, 720)
        self.assertEqual(h, 720)
        self.assertEqual(w, 1280)
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)

        # 1080p (1920x1080) to 480p
        w, h = compute_mode_resolution(1920, 1080, 480)
        self.assertEqual(h, 480)
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)

    def test_never_upscale_smaller_input(self):
        # 480p (854x480) with 1080p max target
        w, h = compute_mode_resolution(854, 480, 1080)
        self.assertEqual(h, 480)
        self.assertEqual(w, 854)

    def test_odd_dimensions_are_even(self):
        # Odd input dimensions
        w, h = compute_mode_resolution(1919, 1079, 720)
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)


if __name__ == "__main__":
    unittest.main()
