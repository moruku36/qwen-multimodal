from pathlib import Path

from qmc.backends import llama_server
from qmc.config import load_config
from qmc.gpu_manager import PROFILES


def test_resolve_files_uses_separate_mmproj_repo(monkeypatch):
    calls = []

    def fake_download(repo, filename, cache):
        calls.append((repo, filename))
        return Path(filename)

    monkeypatch.setattr(llama_server, "download_hf_file", fake_download)
    cfg = load_config()
    model = llama_server.LlamaServerModel(cfg, PROFILES["cpu"])
    model.resolve_files()
    assert calls == [
        (cfg.chat.hf_repo, cfg.chat.model_file),
        (cfg.chat.mmproj_repo, cfg.chat.mmproj_file),
    ]
    assert "abliterated" in model.label
    model.resolve_files()
    assert len(calls) == 2
