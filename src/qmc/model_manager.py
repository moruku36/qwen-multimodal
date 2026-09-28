"""Lazy model lifecycle management with VRAM-aware swapping and OOM recovery.

The manager does not know what a model *is*; it only drives objects implementing
``ManagedModel``. This keeps the UI and engines decoupled from Colab specifics: a remote
backend (RunPod / vLLM) is simply a ManagedModel whose ``uses_local_gpu`` is False.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

from .gpu_manager import GPUProfile, free_cuda_memory, is_oom_error, memory_snapshot

log = logging.getLogger(__name__)

T = TypeVar("T")

CHAT = "chat"
IMAGE = "image"


@runtime_checkable
class ManagedModel(Protocol):
    name: str

    @property
    def is_loaded(self) -> bool: ...

    @property
    def uses_local_gpu(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def degrade(self) -> bool:
        """Switch to a lower-VRAM configuration after an OOM. Return True if anything changed."""
        ...


class ModelLoadError(RuntimeError):
    pass


@dataclass
class ModelEvent:
    ts: float
    model: str
    action: str
    detail: str = ""
    seconds: float | None = None


@dataclass
class ModelManager:
    profile: GPUProfile
    models: dict[str, ManagedModel] = field(default_factory=dict)
    on_status: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self.events: list[ModelEvent] = []
        self.active: str | None = None

    # ------------------------------------------------------------------ registry
    def register(self, model: ManagedModel) -> None:
        self.models[model.name] = model

    def get(self, name: str) -> ManagedModel:
        if name not in self.models:
            raise KeyError(f"Model {name!r} is not registered")
        return self.models[name]

    def loaded_models(self) -> list[str]:
        return [n for n, m in self.models.items() if m.is_loaded]

    def is_loaded(self, name: str) -> bool:
        return name in self.models and self.models[name].is_loaded

    # ------------------------------------------------------------------ planning
    def plan_activation(self, name: str) -> list[str]:
        """Return the models that must be unloaded before ``name`` can use the GPU.

        Pure function of the current state + profile (unit-tested).
        """
        target = self.get(name)
        if self.profile.coresident or not target.uses_local_gpu:
            return []
        return [
            other
            for other, model in self.models.items()
            if other != name and model.is_loaded and model.uses_local_gpu
        ]

    # ------------------------------------------------------------------ lifecycle
    def _status(self, text: str) -> None:
        log.info(text)
        if self.on_status:
            with contextlib.suppress(Exception):  # UI callbacks must never break model management
                self.on_status(text)

    def _record(self, model: str, action: str, detail: str = "", seconds: float | None = None) -> None:
        self.events.append(ModelEvent(time.time(), model, action, detail, seconds))
        del self.events[:-200]

    def ensure(self, name: str) -> ManagedModel:
        """Make ``name`` ready, unloading conflicting models first. Idempotent (no needless reloads)."""
        with self._lock:
            model = self.get(name)
            for other in self.plan_activation(name):
                self.unload(other, reason=f"make room for {name}")
            if not model.is_loaded:
                self._status(f"Loading {name} model...")
                t0 = time.time()
                try:
                    model.load()
                except Exception as exc:
                    self._record(name, "load_failed", repr(exc))
                    free_cuda_memory()
                    if is_oom_error(exc):
                        return self._retry_load_after_oom(model, exc)
                    raise ModelLoadError(f"{name} のロードに失敗しました: {exc}") from exc
                dt = time.time() - t0
                self._record(name, "load", memory_snapshot().summary(), dt)
                self._status(f"{name} loaded in {dt:.0f}s")
            self.active = name
            return model

    def _retry_load_after_oom(self, model: ManagedModel, exc: BaseException) -> ManagedModel:
        self._status(f"CUDA OOM while loading {model.name}; unloading everything else and retrying")
        for other in list(self.models):
            if other != model.name:
                self.unload(other, reason="OOM recovery")
        model.degrade()
        free_cuda_memory()
        try:
            model.load()
        except Exception as exc2:
            raise ModelLoadError(f"{model.name} をロードできません（VRAM不足）: {exc2}") from exc
        self._record(model.name, "load", "after OOM recovery")
        self.active = model.name
        return model

    def unload(self, name: str, reason: str = "") -> None:
        with self._lock:
            model = self.get(name)
            if not model.is_loaded:
                return
            self._status(f"Unloading {name} ({reason})" if reason else f"Unloading {name}")
            t0 = time.time()
            try:
                model.unload()
            finally:
                free_cuda_memory()
            self._record(name, "unload", reason, time.time() - t0)
            if self.active == name:
                self.active = None

    def unload_all(self) -> None:
        with self._lock:
            for name in list(self.models):
                self.unload(name, reason="unload all")

    # ------------------------------------------------------------------ execution
    def run(self, name: str, fn: Callable[[ManagedModel], T], retries: int = 1) -> T:
        """Run ``fn(model)`` with ``name`` active. On CUDA OOM: unload others -> clear cache ->
        degrade (lower-VRAM settings) -> retry. The app never dies on OOM; the final error is
        raised as a normal exception for the UI to display."""
        with self._lock:
            attempt = 0
            while True:
                model = self.ensure(name)
                try:
                    return fn(model)
                except Exception as exc:
                    if not is_oom_error(exc) or attempt >= retries:
                        if is_oom_error(exc):
                            free_cuda_memory()
                            raise RuntimeError(
                                "GPUメモリ不足（CUDA OOM）です。解像度やステップ数を下げて再試行してください。"
                            ) from exc
                        raise
                    attempt += 1
                    self._record(name, "oom", repr(exc)[:300])
                    self._status(f"CUDA OOM in {name}: recovering (attempt {attempt}/{retries})")
                    for other in list(self.models):
                        if other != name:
                            self.unload(other, reason="OOM recovery")
                    free_cuda_memory()
                    changed = model.degrade()
                    if changed:
                        self._status(f"{name}: switched to lower-VRAM settings")

    @contextlib.contextmanager
    def use(self, name: str) -> Iterator[ManagedModel]:
        """Hold the GPU lock while ``name`` is active (used for streaming responses)."""
        with self._lock:
            yield self.ensure(name)

    def status(self) -> dict:
        return {
            "profile": self.profile.key,
            "mode": self.profile.mode,
            "loaded": self.loaded_models(),
            "active": self.active,
        }
