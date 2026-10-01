from pathlib import Path

import pytest

from qmc.config import load_config


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("QMC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("QMC_MOCK", "1")
    monkeypatch.setenv("QMC_GPU_PROFILE", "l4")
    monkeypatch.setenv("QMC_SHARE", "false")
    monkeypatch.setenv("QMC_CHAT_BASE_URL", "http://example:8000/v1")
    monkeypatch.setenv("QMC_CONTENT_POLICY", "standard")
    monkeypatch.setenv("QMC_SEARCH_SAFESEARCH", "off")
    monkeypatch.setenv("QMC_CHAT_HF_REPO", "owner/custom-gguf")
    monkeypatch.setenv("QMC_CHAT_MMPROJ_REPO", "owner/custom-mmproj")
    monkeypatch.setenv("QMC_CHAT_REMOTE_MODEL_NAME", "custom-model")
    cfg = load_config()
    assert cfg.data_dir == tmp_path
    assert cfg.mock is True
    assert cfg.gpu_profile_override == "l4"
    assert cfg.share is False
    assert cfg.chat.remote_base_url == "http://example:8000/v1"
    assert cfg.content_policy == "standard"
    assert cfg.search_safesearch == "off"
    assert cfg.chat.hf_repo == "owner/custom-gguf"
    assert cfg.chat.mmproj_repo == "owner/custom-mmproj"
    assert cfg.chat.remote_model_name == "custom-model"


def test_paths(tmp_path):
    cfg = load_config(data_dir=tmp_path)
    assert cfg.image_dir("abc") == tmp_path / "sessions" / "abc" / "images"
    assert cfg.db_mirror_path == tmp_path / "history.db"


def test_model_and_search_defaults(monkeypatch):
    for name in ("QMC_WEB_SEARCH", "QMC_CHAT_HF_REPO", "QMC_CHAT_MODEL_FILE", "QMC_CHAT_MMPROJ_REPO"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config()
    assert cfg.web_search == "on"
    assert cfg.chat.hf_repo == "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
    assert cfg.chat.model_file == "Huihui-Qwen3.8-27B-abliterated-UD-DW-Q8_K_L.gguf"
    assert cfg.chat.mmproj_repo == "ggml-org/Qwen3.8-27B-GGUF"
    assert cfg.search_max_results == 8
    assert cfg.search_fetch_pages == 5
    assert cfg.search_page_chars == 4000


def test_search_window_env_overrides(monkeypatch):
    monkeypatch.setenv("QMC_SEARCH_MAX_RESULTS", "10")
    monkeypatch.setenv("QMC_SEARCH_FETCH_PAGES", "6")
    monkeypatch.setenv("QMC_SEARCH_PAGE_CHARS", "5000")
    monkeypatch.setenv("QMC_SEARCH_REGION", "wt-wt")
    cfg = load_config()
    assert (cfg.search_max_results, cfg.search_fetch_pages, cfg.search_page_chars, cfg.search_region) == (
        10,
        6,
        5000,
        "wt-wt",
    )


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


def test_image_workload_environment(monkeypatch):
    monkeypatch.setenv("QMC_IMAGE_BAND", "768")
    monkeypatch.setenv("QMC_IMAGE_STEPS", "20")
    monkeypatch.setenv("QMC_IMAGE_MAX_BAND", "1536")
    cfg = load_config()
    assert (cfg.image.default_band, cfg.image.default_steps, cfg.image.max_band) == (768, 20, 1536)
