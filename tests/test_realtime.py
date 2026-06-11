import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.app.realtime.stream_processor import RealTimeProcessor
from face_swap import QualityMode

@pytest.fixture
def mock_processor():
    # Mock FaceSwapper completely so it doesn't try to load ONNX models
    with patch("backend.app.realtime.stream_processor.FaceSwapper") as mock_swapper:
        mock_instance = mock_swapper.return_value
        mock_instance.get_source_face.return_value = "dummy_face"
        
        processor = RealTimeProcessor("dummy_path.jpg")
        
        # Override the app get to return a dummy face
        dummy_detected_face = MagicMock()
        dummy_detected_face.bbox = [0, 0, 100, 100]
        processor.swapper._app.get.return_value = [dummy_detected_face]
        
        # Override swap adapter
        processor.swapper._swap_adapter = MagicMock()
        processor.swapper._swap_adapter.swap_face.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return processor

def test_realtime_processor_latency_adjustment(mock_processor):
    # Set initial quality high
    mock_processor.quality = QualityMode.HIGH
    
    # Simulate a slow frame
    mock_processor.last_process_time = 0.200 # 200ms latency, above 150ms threshold
    
    # Trigger quality adjustment
    mock_processor._adjust_quality()
    
    # Should have adjusted down
    assert mock_processor.quality == QualityMode.FAST

def test_realtime_processor_returns_frame(mock_processor):
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    res = mock_processor.process_frame(dummy_frame)
    
    assert isinstance(res, np.ndarray)
    assert mock_processor.frames_processed == 1
    assert mock_processor.swapper._swap_adapter.swap_face.called

def test_realtime_processor_downscales_huge_frames(mock_processor):
    # Frame larger than 480p height
    dummy_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Mock the return so it matches resize shape
    mock_processor.swapper._swap_adapter.swap_face.return_value = np.zeros((480, 853, 3), dtype=np.uint8)
    
    res = mock_processor.process_frame(dummy_frame)
    
    # Assert it upscales back to the original size
    assert res.shape == (1080, 1920, 3)
