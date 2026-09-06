import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

_BASE_TEST_DIR = Path(__file__).parent / ".test_tmp"
_BASE_TEST_DIR.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(_BASE_TEST_DIR)
os.environ["TEMP"] = str(_BASE_TEST_DIR)
os.environ["TMP"] = str(_BASE_TEST_DIR)


def safe_mkdtemp(*args, **kwargs):
    d = _BASE_TEST_DIR / f"tmp_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


tempfile.mkdtemp = safe_mkdtemp


class SafeTemporaryDirectory:
    def __init__(self, *args, **kwargs):
        self.name = str(_BASE_TEST_DIR / f"tmp_{uuid.uuid4().hex[:8]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=True)


tempfile.TemporaryDirectory = SafeTemporaryDirectory


@pytest.fixture
def tmp_path():
    d = _BASE_TEST_DIR / f"test_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
