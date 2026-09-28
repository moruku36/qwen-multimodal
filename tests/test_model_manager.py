import pytest

from qmc.gpu_manager import PROFILES
from qmc.model_manager import ModelLoadError, ModelManager


class FakeModel:
    def __init__(self, name, local=True, oom_times=0, fail_load=False):
        self.name = name
        self._loaded = False
        self._local = local
        self.loads = 0
        self.unloads = 0
        self.degraded = 0
        self.oom_times = oom_times
        self.fail_load = fail_load

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def uses_local_gpu(self):
        return self._local

    def load(self):
        if self.fail_load:
            raise OSError("download failed")
        self.loads += 1
        self._loaded = True

    def unload(self):
        self.unloads += 1
        self._loaded = False

    def degrade(self):
        self.degraded += 1
        return True


class OutOfMemoryError(RuntimeError):
    pass


def make(profile_key):
    mm = ModelManager(profile=PROFILES[profile_key])
    chat, image = FakeModel("chat"), FakeModel("image")
    mm.register(chat)
    mm.register(image)
    return mm, chat, image


def test_low_vram_swaps_models():
    mm, chat, image = make("l4")
    mm.ensure("chat")
    assert mm.loaded_models() == ["chat"]
    mm.ensure("image")
    assert mm.loaded_models() == ["image"]
    assert chat.unloads == 1
    mm.ensure("chat")
    assert mm.loaded_models() == ["chat"]


def test_no_redundant_reload():
    mm, chat, _ = make("l4")
    mm.ensure("chat")
    mm.ensure("chat")
    mm.ensure("chat")
    assert chat.loads == 1


def test_performance_mode_keeps_both_resident():
    mm, chat, image = make("a100_80")
    mm.ensure("chat")
    mm.ensure("image")
    assert set(mm.loaded_models()) == {"chat", "image"}
    assert chat.unloads == 0


def test_remote_backend_never_forces_swap():
    mm = ModelManager(profile=PROFILES["l4"])
    remote_chat, image = FakeModel("chat", local=False), FakeModel("image")
    mm.register(remote_chat)
    mm.register(image)
    mm.ensure("chat")
    assert mm.plan_activation("image") == []
    mm.ensure("image")
    assert set(mm.loaded_models()) == {"chat", "image"}


def test_run_recovers_from_oom():
    mm, chat, image = make("a100_80")
    mm.ensure("chat")
    calls = {"n": 0}

    def job(model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OutOfMemoryError("CUDA out of memory")
        return "ok"

    assert mm.run("image", job) == "ok"
    assert chat.unloads == 1  # everything else was unloaded
    assert image.degraded == 1


def test_run_gives_friendly_error_after_repeated_oom():
    mm, _, _ = make("l4")

    def job(model):
        raise OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="OOM"):
        mm.run("image", job, retries=1)


def test_non_oom_errors_propagate():
    mm, _, _ = make("l4")

    def job(model):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        mm.run("chat", job)


def test_load_failure_is_wrapped():
    mm = ModelManager(profile=PROFILES["l4"])
    mm.register(FakeModel("chat", fail_load=True))
    with pytest.raises(ModelLoadError):
        mm.ensure("chat")
    assert mm.events[-1].action == "load_failed"
