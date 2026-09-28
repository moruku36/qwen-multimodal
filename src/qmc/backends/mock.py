"""CPU-only fake backends for development, unit tests and UI E2E runs without a GPU.

They exercise the full app path (routing, history, image lineage, UI) but produce placeholder
output. Enabled with ``--mock`` / ``QMC_MOCK=1``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator

from PIL import Image, ImageDraw, ImageEnhance

from .base import Cancelled, ChatDelta, ChatParams, ImageRequest, ProgressFn


class _MockManaged:
    def __init__(self, name: str, load_delay: float = 0.0):
        self.name = name
        self._loaded = False
        self.load_delay = load_delay
        self.load_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def uses_local_gpu(self) -> bool:
        return True

    def load(self) -> None:
        time.sleep(self.load_delay)
        self.load_count += 1
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def degrade(self) -> bool:
        return False


class MockChatModel(_MockManaged):
    label = "MockChat (CPU)"

    def __init__(self, load_delay: float = 0.0, token_delay: float = 0.0):
        super().__init__("chat", load_delay)
        self.token_delay = token_delay
        self.last_messages: list[dict] = []

    def stream_chat(
        self, messages: list[dict], params: ChatParams, cancel: threading.Event | None = None
    ) -> Iterator[ChatDelta]:
        self.last_messages = messages
        last = messages[-1]
        content = last["content"]
        n_images = 0
        text = content
        if isinstance(content, list):
            n_images = sum(1 for p in content if p.get("type") == "image_url")
            text = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        if "Qwen-Image-2.1" in text and "Output ONLY the prompt" in text:
            request = text.rsplit("User request:", 1)[-1].strip()
            yield ChatDelta(content=f"[rewritten] {request}")
            return
        if params.thinking:
            yield ChatDelta(reasoning="(mock) ユーザーの質問を整理しています…")
        reply = f"（モック応答）受け取ったメッセージ: 「{text.strip()[-200:]}」"
        if n_images:
            reply += f"\n画像 {n_images} 枚を受け取りました（モックなので内容は解析していません）。"
        for chunk in _chunks(reply, 8):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if self.token_delay:
                time.sleep(self.token_delay)
            yield ChatDelta(content=chunk)
        yield ChatDelta(finish_reason="stop")


class MockImageModel(_MockManaged):
    label = "MockImage (CPU)"

    def __init__(self, load_delay: float = 0.0, step_delay: float = 0.0):
        super().__init__("image", load_delay)
        self.step_delay = step_delay
        self.requests: list[ImageRequest] = []

    def generate(
        self, request: ImageRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> Image.Image:
        self.requests.append(request)
        for step in range(1, request.steps + 1):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if self.step_delay:
                time.sleep(self.step_delay)
            if progress:
                progress(step, request.steps)
        w = request.width or 512
        h = request.height or 512
        if request.images:
            src = request.images[-1].convert("RGB").resize((w, h))
            img = ImageEnhance.Brightness(src).enhance(0.85)
        else:
            digest = hashlib.sha256(f"{request.prompt}{request.seed}".encode()).digest()
            img = Image.new("RGB", (w, h), tuple(digest[:3]))
        draw = ImageDraw.Draw(img)
        label = ("EDIT" if request.is_edit else "GEN") + f" seed={request.seed}"
        draw.rectangle([0, 0, w, 28], fill=(0, 0, 0))
        draw.text((6, 8), label, fill=(255, 255, 255))
        draw.text((6, h - 20), request.prompt[:60], fill=(255, 255, 255))
        return img


def _chunks(text: str, n: int) -> Iterator[str]:
    for i in range(0, len(text), n):
        yield text[i : i + n]
