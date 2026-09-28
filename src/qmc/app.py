"""Composition root: wires config -> GPU profile -> backends -> engines -> controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chat_engine import ChatEngine
from .config import AppConfig
from .controller import ChatController
from .gpu_manager import GPUInfo, GPUProfile, detect_gpu, select_profile
from .history_manager import HistoryStore
from .image_engine import ImageEngine
from .model_manager import ModelManager
from .session_manager import SessionManager
from .vision_engine import VisionEngine

log = logging.getLogger(__name__)


@dataclass
class App:
    cfg: AppConfig
    gpu: GPUInfo
    profile: GPUProfile
    store: HistoryStore
    sessions: SessionManager
    manager: ModelManager
    controller: ChatController

    @property
    def chat_label(self) -> str:
        return getattr(self.manager.get("chat"), "label", "chat")

    @property
    def image_label(self) -> str:
        return getattr(self.manager.get("image"), "label", "image")


def build_app(cfg: AppConfig, gpu: GPUInfo | None = None) -> App:
    gpu = gpu or detect_gpu()
    profile = select_profile(
        gpu, "cpu" if cfg.mock and not cfg.gpu_profile_override else cfg.gpu_profile_override
    )
    log.info("GPU: %s (%.1f GiB) -> profile %s / %s", gpu.name, gpu.total_gib, profile.key, profile.mode)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    mirror = cfg.db_mirror_path if cfg.db_mirror_path.resolve() != cfg.local_db_path.resolve() else None
    store = HistoryStore(cfg.local_db_path, mirror)
    sessions = SessionManager(store, cfg.data_dir)

    manager = ModelManager(profile=profile)
    if cfg.mock:
        from .backends.mock import MockChatModel, MockImageModel

        manager.register(MockChatModel(load_delay=0.3, token_delay=0.01))
        manager.register(MockImageModel(load_delay=0.3, step_delay=0.02))
    else:
        from .backends.llama_server import LlamaServerModel, RemoteChatModel
        from .backends.qwen_image import QwenImageModel

        chat_model = RemoteChatModel(cfg) if cfg.chat.remote_base_url else LlamaServerModel(cfg, profile)
        manager.register(chat_model)
        manager.register(QwenImageModel(cfg, profile))

    chat = ChatEngine(manager, cfg.max_context_messages, cfg.max_context_images)
    controller = ChatController(
        sessions,
        manager,
        chat,
        VisionEngine(chat),
        ImageEngine(manager, profile, cfg.image.default_band),
        max_image_side=cfg.max_image_side,
        max_upload_mb=cfg.max_upload_mb,
        after_turn=store.sync,
    )
    return App(cfg, gpu, profile, store, sessions, manager, controller)
