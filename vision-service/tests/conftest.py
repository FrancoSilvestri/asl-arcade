import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl_vision import api as api_module  # noqa: E402


@pytest.fixture
def client():
    api_module.app.config.update(TESTING=True)
    return api_module.app.test_client()


@pytest.fixture
def api():
    """The api module, so tests can monkeypatch the detector it calls into."""
    return api_module
