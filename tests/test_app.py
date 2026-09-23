"""Smoke test of the Streamlit app (no file upload; AppTest cannot simulate one)."""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("keras")
from streamlit.testing.v1 import AppTest  # noqa: E402

from skin_detection.config import PROJECT_ROOT  # noqa: E402

APP = str(PROJECT_ROOT / "app.py")


def test_app_loads_model_and_shows_disclaimer():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    text = " ".join(str(w.value) for w in at.warning) + " ".join(str(c.value) for c in at.caption)
    assert "not a medical diagnosis" in text


def test_app_shows_error_instead_of_untrained_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIN_MODEL_PATH", str(tmp_path / "missing.keras"))
    monkeypatch.setenv("SKIN_MODEL_META_PATH", str(tmp_path / "missing.json"))
    import importlib

    from skin_detection import config
    importlib.reload(config)
    try:
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert at.error and "could not be loaded" in at.error[0].value
        assert not at.file_uploader  # app stopped before offering predictions
    finally:
        monkeypatch.undo()
        importlib.reload(config)
