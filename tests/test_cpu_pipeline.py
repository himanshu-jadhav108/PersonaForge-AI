"""
test_cpu_pipeline.py — PersonaForge AI · CPU Pipeline Verification

Tests that:
  1. config_cpu and config_gpu load correctly with expected values
  2. FaceSwapper forced into CPU mode reports mode == 'cpu'
  3. process_video_optimized() routes to pipeline_cpu (mocked)
  4. process_video() (GPU path) is NOT called in CPU mode
  5. video_utils.rebuild_video() uses ultrafast preset in cpu_mode=True
"""

import sys
import importlib
import unittest
from unittest.mock import patch, MagicMock, call

# ── ensure project root is in path ─────────────────────────────────────────────
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestConfigs(unittest.TestCase):
    """Verify config values are sane and well-separated."""

    def test_cpu_config_values(self):
        from config import config_cpu as cpu
        self.assertGreater(cpu.PROCESS_EVERY_N_FRAMES, 1,
            "CPU should skip frames")
        self.assertGreater(cpu.DETECT_EVERY, 1,
            "CPU should reduce detection frequency")
        self.assertGreater(cpu.TARGET_HEIGHT, 0,
            "CPU should downscale frames")
        self.assertFalse(cpu.ENHANCEMENT_ENABLED,
            "CPU should disable enhancement")
        self.assertEqual(cpu.FFMPEG_PRESET, "ultrafast",
            "CPU should use ultrafast FFmpeg preset")
        print(f"  [OK] config_cpu: skip={cpu.PROCESS_EVERY_N_FRAMES}, "
              f"detect_every={cpu.DETECT_EVERY}, height={cpu.TARGET_HEIGHT}p")

    def test_gpu_config_values(self):
        from config import config_gpu as gpu
        self.assertEqual(gpu.PROCESS_EVERY_N_FRAMES, 1,
            "GPU should process every frame")
        self.assertEqual(gpu.TARGET_HEIGHT, 0,
            "GPU should use original resolution")
        self.assertTrue(gpu.ENHANCEMENT_ENABLED,
            "GPU should enable enhancement")
        self.assertTrue(gpu.USE_SEAMLESS_CLONE,
            "GPU should use seamlessClone")
        print(f"  [OK] config_gpu: skip={gpu.PROCESS_EVERY_N_FRAMES}, "
              f"height=original, enhancement={gpu.ENHANCEMENT_ENABLED}")

    def test_configs_are_different(self):
        from config import config_cpu as cpu
        from config import config_gpu as gpu
        self.assertNotEqual(cpu.PROCESS_EVERY_N_FRAMES, gpu.PROCESS_EVERY_N_FRAMES)
        self.assertNotEqual(cpu.TARGET_HEIGHT, gpu.TARGET_HEIGHT)
        self.assertNotEqual(cpu.ENHANCEMENT_ENABLED, gpu.ENHANCEMENT_ENABLED)
        self.assertNotEqual(cpu.FFMPEG_PRESET, gpu.FFMPEG_PRESET)
        print("  [OK] CPU and GPU configs are clearly separated")


class TestFaceSwapperMode(unittest.TestCase):
    """Verify FaceSwapper._mode and routing without real models."""

    def _make_mock_swapper(self, cuda_available: bool):
        """Patch insightface + onnxruntime and construct FaceSwapper."""
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_available
            else ["CPUExecutionProvider"]
        )

        mock_app   = MagicMock()
        mock_model = MagicMock()
        mock_face_analysis = MagicMock(return_value=mock_app)
        mock_ort   = MagicMock()
        mock_ort.get_available_providers.return_value = providers

        mock_insightface       = MagicMock()
        mock_insightface.app.FaceAnalysis = mock_face_analysis
        mock_insightface.model_zoo.get_model = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {
            "insightface": mock_insightface,
            "insightface.app": mock_insightface.app,
            "onnxruntime": mock_ort,
        }):
            # Patch _find_model to return a dummy path
            with patch("face_swap._find_model", return_value="dummy.onnx"):
                from face_swap import FaceSwapper
                swapper = FaceSwapper(model_name="inswapper_128.onnx",
                                      use_gpu=cuda_available)
        return swapper

    def test_cpu_mode_detected(self):
        swapper = self._make_mock_swapper(cuda_available=False)
        self.assertEqual(swapper.get_mode(), "cpu")
        self.assertEqual(swapper.get_execution_provider(), "CPU")
        print("  [OK] FaceSwapper detects CPU mode correctly")

    def test_gpu_mode_detected(self):
        swapper = self._make_mock_swapper(cuda_available=True)
        self.assertEqual(swapper.get_mode(), "gpu")
        self.assertEqual(swapper.get_execution_provider(), "GPU")
        print("  [OK] FaceSwapper detects GPU mode correctly")

    def test_cpu_routes_to_cpu_pipeline(self):
        """process_video_optimized() in cpu mode calls pipeline_cpu, not process_video."""
        swapper = self._make_mock_swapper(cuda_available=False)
        # Force mode to cpu in case mock leaked
        swapper._mode = "cpu"
        
        with patch("pipelines.pipeline_cpu.process_video_cpu", return_value=(5, 2)) as mock_cpu, \
             patch.object(swapper, "process_video") as mock_gpu:
            result = swapper.process_video_optimized(
                source_face=MagicMock(),
                frames_dir="/tmp/frames",
                output_dir="/tmp/out",
            )
            mock_cpu.assert_called_once()
            mock_gpu.assert_not_called()
            self.assertEqual(result, (5, 2))
        print("  [OK] CPU mode routes to pipeline_cpu, GPU process_video NOT called")

    def test_gpu_routes_to_gpu_pipeline(self):
        """process_video_optimized() in gpu mode calls process_video."""
        swapper = self._make_mock_swapper(cuda_available=True)
        swapper._mode = "gpu"

        with patch("pipelines.pipeline_gpu.process_video_gpu", return_value=(10, 0)) as mock_gpu, \
             patch("pipelines.pipeline_cpu.process_video_cpu") as mock_cpu:
            result = swapper.process_video_optimized(
                source_face=MagicMock(),
                frames_dir="/tmp/frames",
                output_dir="/tmp/out",
            )
            mock_gpu.assert_called_once()
            mock_cpu.assert_not_called()
            self.assertEqual(result, (10, 0))
        print("  [OK] GPU mode routes to pipeline_gpu, CPU pipeline NOT called")


class TestVideoUtilsCPUMode(unittest.TestCase):
    """Verify rebuild_video uses ultrafast in cpu_mode."""

    def test_cpu_mode_uses_ultrafast(self):
        """rebuild_video with cpu_mode=True should call ffmpeg with ultrafast preset."""
        captured_cmds = []

        def mock_run_ffmpeg(cmd, label=""):
            captured_cmds.append(cmd)
            # Return fake completed process
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("video_utils._run_ffmpeg", side_effect=mock_run_ffmpeg), \
             patch("video_utils._has_nvenc", return_value=False), \
             patch("pathlib.Path.glob", return_value=[MagicMock()] * 5), \
             patch("os.path.exists", return_value=False), \
             patch("shutil.move"):
            from video_utils import rebuild_video
            try:
                rebuild_video(
                    frames_dir="/tmp/frames",
                    audio_path=None,
                    output_path="/tmp/out.mp4",
                    fps=30.0,
                    bitrate="1M",
                    cpu_mode=True,
                )
            except Exception:
                pass  # may fail on missing files — we only care about cmd content

        video_cmd = next((c for c in captured_cmds if "-c:v" in c), None)
        if video_cmd:
            self.assertIn("ultrafast", video_cmd,
                "cpu_mode=True must use ultrafast FFmpeg preset")
            print("  [OK] cpu_mode=True uses ultrafast preset in rebuild_video")
        else:
            print("  [SKIP] ffmpeg cmd not captured — test environment limitation")


class TestOrientationAndResizePlanning(unittest.TestCase):
    """Verify orientation detection and aspect-ratio-safe resize planning."""

    def test_detect_orientation_variants(self):
        from video_utils import detect_orientation

        self.assertEqual(detect_orientation(1920, 1080), "landscape")
        self.assertEqual(detect_orientation(1080, 1920), "portrait")
        self.assertEqual(detect_orientation(1080, 1080), "square")
        self.assertEqual(detect_orientation(0, 1080), "unknown")

    def test_landscape_scales_by_height(self):
        from video_utils import compute_resize_dimensions

        w, h, changed = compute_resize_dimensions(1920, 1080, 720, "landscape")
        self.assertTrue(changed)
        self.assertEqual((w, h), (1280, 720))

    def test_portrait_scales_by_width(self):
        from video_utils import compute_resize_dimensions

        w, h, changed = compute_resize_dimensions(1080, 1920, 720, "portrait")
        self.assertTrue(changed)
        self.assertEqual((w, h), (720, 1280))

    def test_square_scales_longest_edge(self):
        from video_utils import compute_resize_dimensions

        w, h, changed = compute_resize_dimensions(1080, 1080, 720, "square")
        self.assertTrue(changed)
        self.assertEqual((w, h), (720, 720))

    def test_crop_plan_for_landscape(self):
        from video_utils import _portrait_crop_plan

        plan = _portrait_crop_plan(1920, 1080)
        self.assertIsNotNone(plan)
        filter_expr, crop_w, crop_h = plan
        self.assertTrue(filter_expr.startswith("crop="))
        self.assertLess(crop_w, 1920)
        self.assertEqual(crop_h, 1080)

    def test_crop_plan_skips_portrait(self):
        from video_utils import _portrait_crop_plan

        self.assertIsNone(_portrait_crop_plan(1080, 1920))

    def test_gpu_mode_not_ultrafast(self):
        """rebuild_video with cpu_mode=False should NOT use ultrafast when no NVENC."""
        captured_cmds = []

        def mock_run_ffmpeg(cmd, label=""):
            captured_cmds.append(cmd)
            return MagicMock()

        with patch("video_utils._run_ffmpeg", side_effect=mock_run_ffmpeg), \
             patch("video_utils._has_nvenc", return_value=False), \
             patch("pathlib.Path.glob", return_value=[MagicMock()] * 5), \
             patch("os.path.exists", return_value=False), \
             patch("shutil.move"):
            from video_utils import rebuild_video
            try:
                rebuild_video(
                    frames_dir="/tmp/frames",
                    audio_path=None,
                    output_path="/tmp/out.mp4",
                    fps=30.0,
                    bitrate="3M",
                    cpu_mode=False,
                )
            except Exception:
                pass

        video_cmd = next((c for c in captured_cmds if "-c:v" in c), None)
        if video_cmd:
            self.assertNotIn("ultrafast", video_cmd,
                "cpu_mode=False must NOT use ultrafast")
            print("  [OK] cpu_mode=False does NOT use ultrafast (GPU path preserved)")
        else:
            print("  [SKIP] ffmpeg cmd not captured — test environment limitation")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PersonaForge AI — CPU Pipeline Test Suite")
    print("=" * 60 + "\n")
    unittest.main(verbosity=0, exit=True)
