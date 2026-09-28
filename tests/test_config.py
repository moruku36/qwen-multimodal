from pathlib import Path

import pytest

from qmc.config import load_config


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("QMC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QMC_MOCK", "1")
    monkeypatch.setenv("QMC_GPU_PROFILE", "l4")
    monkeypatch.setenv("QMC_SHARE", "false")
    monkeypatch.setenv("QMC_CHAT_BASE_URL", "http://example:8000/v1")
    cfg = load_config()
    assert cfg.data_dir == tmp_path
    assert cfg.mock is True
    assert cfg.gpu_profile_override == "l4"
    assert cfg.share is False
    assert cfg.chat.remote_base_url == "http://example:8000/v1"


def test_paths(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    assert cfg.image_dir("abc") == tmp_path / "sessions" / "abc" / "images"
    assert cfg.db_mirror_path == tmp_path / "history.db"


def test_public_dict_hides_secrets(monkeypatch):
    monkeypatch.setenv("QMC_AUTH_PASSWORD", "secret-pw")
    monkeypatch.setenv("QMC_CHAT_API_KEY", "secret-key")
    cfg = load_config()
    text = str(cfg.public_dict())
    assert "secret-pw" not in text
    assert "secret-key" not in text


def test_unknown_override_rejected():
    with pytest.raises(AttributeError):
        load_config(nonexistent=1)


def test_default_data_dir_outside_colab(monkeypatch):
    monkeypatch.delenv("QMC_DATA_DIR", raising=False)
    cfg = load_config()
    assert isinstance(cfg.data_dir, Path)
