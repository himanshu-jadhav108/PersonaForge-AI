"""
tests/test_phase0_fixes.py — Regression Test Suite for Phase 0 Critical Fixes
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# If cv2 or plotly is not installed in the test runner environment, mock them
if "cv2" not in sys.modules:
    try:
        import cv2
    except ImportError:
        sys.modules["cv2"] = MagicMock()

if "plotly" not in sys.modules:
    try:
        import plotly
    except ImportError:
        plotly_mock = MagicMock()
        sys.modules["plotly"] = plotly_mock
        sys.modules["plotly.graph_objects"] = MagicMock()
        sys.modules["plotly.express"] = MagicMock()

if "python_multipart" not in sys.modules:
    try:
        import python_multipart
    except ImportError:
        sys.modules["python_multipart"] = MagicMock()
        sys.modules["multipart"] = MagicMock()

try:
    import fastapi.dependencies.utils as fdu
    fdu.ensure_multipart_is_installed = lambda: None
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestPhase0Fixes(unittest.TestCase):
    """Verify each Phase 0 issue against the codebase."""

    def test_01_main_imports_cv2(self):
        """Issue 1: main.py must import cv2."""
        main_path = PROJECT_ROOT / "main.py"
        content = main_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(main_path))
        
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
                    
        self.assertIn("cv2", imported_modules, "main.py must import cv2")

    def test_02_no_duplicate_imports_in_main(self):
        """Issue 2: main.py must not have duplicate imports."""
        main_path = PROJECT_ROOT / "main.py"
        content = main_path.read_text(encoding="utf-8")
        
        lines = [line.strip() for line in content.splitlines()]
        import_lines = [l for l in lines if l.startswith(("import ", "from "))]
        
        duplicates = set()
        seen = set()
        for imp in import_lines:
            if imp in seen:
                duplicates.add(imp)
            seen.add(imp)
            
        self.assertEqual(len(duplicates), 0, f"Found duplicate imports in main.py: {duplicates}")

    def test_03_plotly_and_psutil_in_requirements(self):
        """Issue 3: plotly and psutil must be present in requirements.txt."""
        req_path = PROJECT_ROOT / "requirements.txt"
        content = req_path.read_text(encoding="utf-8")
        self.assertRegex(content, r"(?i)plotly\s*([>=<].*)?", "requirements.txt must include plotly")
        self.assertRegex(content, r"(?i)psutil\s*([>=<].*)?", "requirements.txt must include psutil")

    def test_04_ffmpeg_writer_uses_pipe_stderr(self):
        """Issue 4: FFmpeg writer in video_utils.py must capture stderr via PIPE."""
        vutils_path = PROJECT_ROOT / "video_utils.py"
        content = vutils_path.read_text(encoding="utf-8")
        self.assertIn("stderr=subprocess.PIPE", content, "FFmpeg writer should use stderr=subprocess.PIPE")

    def test_05_stream_upload_to_disk_exists(self):
        """Issue 5: main.py must define _stream_upload_to_disk to prevent memory exhaustion."""
        from main import _stream_upload_to_disk
        self.assertTrue(inspect.iscoroutinefunction(_stream_upload_to_disk))

    def test_07_stub_adapters_raise_not_implemented(self):
        """Issue 7: SimSwapAdapter and GhostAdapter must raise NotImplementedError."""
        from backend.app.models.adapters.ghost import GhostAdapter
        from backend.app.models.adapters.simswap import SimSwapAdapter
        
        sim = SimSwapAdapter()
        sim.load_model("dummy_path", ["CPUExecutionProvider"])
        with self.assertRaises(NotImplementedError):
            sim.swap_face(None, None, None)
            
        ghost = GhostAdapter()
        ghost.load_model("dummy_path", ["CPUExecutionProvider"])
        with self.assertRaises(NotImplementedError):
            ghost.swap_face(None, None, None)

    def test_08_analytics_router_psutil_resilience(self):
        """Issue 8: analytics router must handle missing psutil gracefully."""
        import asyncio

        from backend.app.analytics.router import get_system_health
        health = asyncio.run(get_system_health())
        self.assertIn("cpu_utilization", health)
        self.assertIn("memory_utilization", health)

    def test_09_realtime_router_fallback(self):
        """Issue 9: realtime router must not crash if aiortc is absent."""
        from backend.app.realtime.router import router
        self.assertIsNotNone(router)

    def test_10_even_dimensions_helper(self):
        """Issue 10: Dimensions passed to video encoder must be even."""
        from video_utils import _even
        self.assertEqual(_even(1079), 1078)
        self.assertEqual(_even(1080), 1080)
        self.assertEqual(_even(1919), 1918)
        self.assertEqual(_even(1), 2)

    def test_11_bitrate_parameter_supported(self):
        """Issue 11: FaceSwapper and CPU pipeline must support bitrate parameter."""
        from face_swap import FaceSwapper
        from pipelines.pipeline_cpu import process_video_cpu
        
        sig_opt = inspect.signature(FaceSwapper.process_video_optimized)
        self.assertIn("bitrate", sig_opt.parameters)
        
        sig_cpu = inspect.signature(process_video_cpu)
        self.assertIn("bitrate", sig_cpu.parameters)

    def test_13_session_id_validation(self):
        """Issue 13: Session IDs must be validated to prevent directory traversal."""
        from main import validate_session_id
        
        valid_id = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
        self.assertTrue(validate_session_id(valid_id))
        
        invalid_ids = [
            "../../etc/passwd",
            "../uploads",
            "a1b2c3",
            "a1b2c3d4e5f60718293a4b5c6d7e8f90/extra",
            "invalid_hex_characters_here!!",
        ]
        for bad_id in invalid_ids:
            self.assertFalse(validate_session_id(bad_id), f"Should reject invalid session_id: {bad_id}")

    def test_14_magic_bytes_validation(self):
        """Issue 14: File magic bytes validation must identify valid vs spoofed media."""
        import tempfile

        from main import validate_media_magic_bytes
        
        # Test JPEG magic bytes (\xFF\xD8\xFF)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01")
            tf_path = Path(tf.name)
            
        try:
            self.assertTrue(validate_media_magic_bytes(tf_path, "image"))
        finally:
            tf_path.unlink(missing_ok=True)
            
        # Test invalid file spoofed as .jpg
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(b"NOT A REAL JPEG FILE CONTENT")
            tf_path = Path(tf.name)
            
        try:
            self.assertFalse(validate_media_magic_bytes(tf_path, "image"))
        finally:
            tf_path.unlink(missing_ok=True)

    def test_15_retention_config_exists(self):
        """Issue 15: Configurable retention hours must exist."""
        from main import RETENTION_HOURS
        self.assertIsInstance(RETENTION_HOURS, int)
        self.assertGreater(RETENTION_HOURS, 0)

    def test_16_model_fallback_mirrors(self):
        """Issue 16: MODEL_CONFIG must include fallback mirrors."""
        from models.model_manager import MODEL_CONFIG
        self.assertIn("inswapper", MODEL_CONFIG)
        cfg = MODEL_CONFIG["inswapper"]
        self.assertIn("urls", cfg, "Model config should provide multiple mirror URLs")
        self.assertGreater(len(cfg["urls"]), 1, "Should have at least 1 primary and 1 fallback mirror")

    def test_17_no_double_warmup_in_main(self):
        """Issue 17: main.py lifespan must not redundantly warm up the swapper."""
        main_path = PROJECT_ROOT / "main.py"
        content = main_path.read_text(encoding="utf-8")
        
        self.assertNotIn("app.state.swapper._warm_up()", content,
                         "Redundant warm-up call should be removed from main.py lifespan")


if __name__ == "__main__":
    unittest.main()
