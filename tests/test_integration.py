import pytest

def test_onnxruntime_providers():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
    except ImportError:
        pytest.skip("onnxruntime not installed in this test environment")