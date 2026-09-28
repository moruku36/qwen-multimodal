from pathlib import Path

from qmc import colab


def test_cmake_command_builds_static_multi_arch():
    cmd = colab.cmake_configure_cmd(Path("/src"), Path("/b"))
    joined = " ".join(cmd)
    assert "-DGGML_CUDA=ON" in joined
    assert "-DCMAKE_CUDA_ARCHITECTURES=80;89;90" in joined
    assert "-DBUILD_SHARED_LIBS=OFF" in joined


def test_cache_dir_is_keyed_by_commit_and_arch(tmp_path):
    d = colab.llama_cache_dir(tmp_path, commit="abcdef1234567", archs="80;89")
    assert d == tmp_path / "abcdef1234-sm80_89"


def test_outside_colab_is_safe(monkeypatch):
    monkeypatch.setattr(colab, "in_colab", lambda: False)
    assert colab.load_secrets() == []
    info = colab.setup(use_drive=False)
    assert info["drive"] is False


def test_notebook_is_thin():
    import json

    nb = json.loads((Path(__file__).parents[1] / "Qwen-Multimodal-Colab.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 4
    assert all(len(c["source"]) < 15 for c in code)
    assert not any("HF_TOKEN=" in "".join(c["source"]) for c in code)  # no secrets in the notebook
