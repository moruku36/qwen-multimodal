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

    nb = json.loads((Path(__file__).parents[1] / "Qwen-Multimodal-Colab.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 5
    assert all(len(c["source"]) <= 15 for c in code)
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
    colab._set_current_app(prev)
    colab.shutdown_previous()
    assert prev.demo.closed and prev.manager.unloaded
    assert colab._get_current_app() is None


def _notebook_code_cells():
    import json

    nb = json.loads((Path(__file__).parents[1] / "Qwen-Multimodal-Colab.ipynb").read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _fake_colab_runtime(monkeypatch, calls, *, flush=None, unassign=None):
    import sys
    from types import ModuleType, SimpleNamespace

    module = ModuleType("google.colab")
    module.drive = SimpleNamespace(
        flush_and_unmount=flush or (lambda **kw: calls.append(("flush", kw["timeout_ms"])))
    )
    module.runtime = SimpleNamespace(unassign=unassign or (lambda: calls.append("unassign")))
    monkeypatch.setitem(sys.modules, "google.colab", module)
    monkeypatch.setattr(colab, "in_colab", lambda: True)
    monkeypatch.setattr(colab, "_get_current_app", lambda: None)
    monkeypatch.setattr(colab, "shutdown_previous", lambda: calls.append("cleanup"))


def test_shutdown_runtime_orders_cleanup_flush_unassign(monkeypatch):
    from types import SimpleNamespace

    calls = []
    _fake_colab_runtime(monkeypatch, calls)
    app = SimpleNamespace(controller=SimpleNamespace(cancel=lambda: calls.append("cancel")))
    monkeypatch.setattr(colab, "_get_current_app", lambda: app)
    colab.shutdown_runtime()
    assert calls == ["cancel", "cleanup", ("flush", 30_000), "unassign"]


def test_shutdown_runtime_still_unassigns_after_cleanup_and_flush_fail(monkeypatch, capsys):
    calls = []

    def fail(**kwargs):
        raise RuntimeError("cleanup failed")

    _fake_colab_runtime(monkeypatch, calls, flush=fail)
    monkeypatch.setattr(colab, "shutdown_previous", fail)
    colab.shutdown_runtime()
    assert calls == ["unassign"]
    assert "未同期" in capsys.readouterr().out


def test_shutdown_runtime_outside_colab_only_cleans_app(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(colab, "in_colab", lambda: False)
    monkeypatch.setattr(colab, "_get_current_app", lambda: None)
    monkeypatch.setattr(colab, "shutdown_previous", lambda: calls.append("cleanup"))
    colab.shutdown_runtime()
    assert calls == ["cleanup"]
    assert "Colab 外" in capsys.readouterr().out


def test_shutdown_runtime_unassign_failure_is_reported(monkeypatch, capsys):
    import pytest

    calls = []

    def fail():
        raise RuntimeError("backend unavailable")

    _fake_colab_runtime(monkeypatch, calls, unassign=fail)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        colab.shutdown_runtime()
    assert calls == ["cleanup", ("flush", 30_000)]
    assert "セッションを終了" in capsys.readouterr().out


def test_shutdown_runtime_bounds_stuck_cleanup(monkeypatch, capsys):
    import threading

    calls = []
    release = threading.Event()
    finished = threading.Event()

    def stuck():
        release.wait()

    def flush(**kwargs):
        finished.set()

    _fake_colab_runtime(monkeypatch, calls, flush=flush)
    monkeypatch.setattr(colab, "shutdown_previous", stuck)
    try:
        colab.shutdown_runtime(cleanup_timeout=0.01)
        assert calls == ["unassign"]
        assert "タイムアウト" in capsys.readouterr().out
    finally:
        release.set()
        assert finished.wait(timeout=2)


def test_shutdown_runtime_unassigns_when_wait_is_interrupted(monkeypatch):
    import pytest

    calls = []
    _fake_colab_runtime(monkeypatch, calls)

    class InterruptedWorker:
        def __init__(self, **kwargs):
            assert kwargs["daemon"] is True

        def start(self):
            pass

        def join(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(colab.threading, "Thread", InterruptedWorker)
    with pytest.raises(KeyboardInterrupt):
        colab.shutdown_runtime()
    assert calls == ["unassign"]


def test_launch_cell_preserves_successful_background_app():
    from types import SimpleNamespace

    calls = []
    app = object()
    fake = SimpleNamespace(launch=lambda **kw: app, shutdown_runtime=lambda: calls.append("shutdown"))
    ns = {"colab": fake, "SHARE": False, "PROFILE": None, "WEB_SEARCH": "on"}
    exec(_notebook_code_cells()[3], ns)
    assert ns["app"] is app
    assert calls == []


def test_launch_cell_cleans_startup_error_and_interrupt_preserving_original():
    from types import SimpleNamespace

    import pytest

    for error in (RuntimeError("launch failed"), KeyboardInterrupt()):
        for cleanup_fails in (False, True):
            calls = []

            def launch(error=error, **kwargs):
                raise error

            def shutdown(calls=calls, cleanup_fails=cleanup_fails):
                calls.append("shutdown")
                if cleanup_fails:
                    raise RuntimeError("unassign failed")

            ns = {
                "colab": SimpleNamespace(launch=launch, shutdown_runtime=shutdown),
                "SHARE": False,
                "PROFILE": None,
                "WEB_SEARCH": "on",
            }
            with pytest.raises(type(error)) as caught:
                exec(_notebook_code_cells()[3], ns)
            assert caught.value is error
            assert calls == ["shutdown"]


def test_shutdown_cell_only_displays_control(monkeypatch):
    calls = []
    monkeypatch.setattr(colab, "show_shutdown_button", lambda: calls.append("display"))
    monkeypatch.setattr(colab, "shutdown_runtime", lambda: calls.append("shutdown"))
    exec(_notebook_code_cells()[4], {})
    assert calls == ["display"]


def test_shutdown_control_requires_fresh_confirmation(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    calls = []
    displayed = []

    class Widget:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.disabled = kwargs.get("disabled", False)
            self._value = kwargs.get("value", False)
            self.observer = None

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, new):
            self._value = new
            if self.observer:
                self.observer({"new": new})

        def observe(self, callback, names):
            self.observer = callback

        def on_click(self, callback):
            self.click = lambda: callback(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    widgets = SimpleNamespace(Checkbox=Widget, Button=Widget, Output=Widget, Layout=Widget, VBox=lambda x: x)
    monkeypatch.setitem(sys.modules, "ipywidgets", widgets)
    ipython = ModuleType("IPython.display")
    ipython.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython.display", ipython)
    monkeypatch.setattr(colab, "shutdown_runtime", lambda: calls.append("shutdown"))
    colab.show_shutdown_button()
    confirm, button, _ = displayed[-1]
    assert not confirm.value and button.disabled
    button.click()
    assert calls == []
    confirm.value = True
    assert not button.disabled
    confirm.value = False
    button.click()
    assert calls == []
    confirm.value = True
    button.click()
    button.click()
    assert calls == ["shutdown"]
    assert confirm.disabled and button.disabled
    colab.show_shutdown_button()
    assert not displayed[-1][0].value and displayed[-1][1].disabled


def test_image_prefetch_reuses_loader_cache(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from qmc import colab
    from qmc.config import load_config

    calls = []
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr("qmc.backends.llama_server.download_hf_file", lambda *a: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    colab.prefetch_models(chat=False)
    assert calls == [((load_config().image.model_id,), {"cache_dir": str(load_config().hf_cache_dir)})]
