from pathlib import Path

from qmc import colab


def test_cmake_command_builds_static_multi_arch():
    cmd = colab.cmake_configure_cmd(Path("/src"), Path("/b"), "80")
    joined = " ".join(cmd)
    assert "-DGGML_CUDA=ON" in joined
    assert "-DCMAKE_CUDA_ARCHITECTURES=80" in joined
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


def test_parse_compute_cap():
    assert colab.parse_compute_cap("8.0") == "80"
    assert colab.parse_compute_cap("8.9\n") == "89"
    assert colab.parse_compute_cap("") is None
    assert colab.parse_compute_cap("N/A") is None


def test_cuda_archs_env_override(monkeypatch):
    monkeypatch.setenv("QMC_CUDA_ARCHS", "90")
    assert colab.cuda_archs() == "90"


def test_find_free_port_skips_busy_port():
    import socket

    with socket.socket() as busy:
        busy.bind(("0.0.0.0", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert not colab.port_is_free(port)
        assert colab.find_free_port(port) != port


def test_shutdown_previous_closes_and_unloads():
    class Demo:
        closed = False

        def close(self):
            self.closed = True

    class Manager:
        unloaded = False

        def unload_all(self):
            self.unloaded = True

    class Store:
        def sync(self):
            pass

        def close(self):
            pass

    class App:
        demo = Demo()
        manager = Manager()
        store = Store()

    prev = App()
    colab._CURRENT_APP = prev
    colab.shutdown_previous()
    assert prev.demo.closed and prev.manager.unloaded
    assert colab._CURRENT_APP is None
