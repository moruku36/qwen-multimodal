"""Vision: questions about uploaded / generated images, and before/after comparison.

Qwen3.8-27B is a native VLM (mmproj loaded into llama-server), so Vision reuses the chat model;
no separate VL model is kept in VRAM (ADR-0001).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

from .backends.base import ChatDelta, ChatParams
from .chat_engine import ChatEngine, ContextImage, ContextMessage

COMPARE_HINT = (
    "（システム補足）次の{n}枚の画像を順番に比較してください。1枚目が「{first}」、最後が「{last}」です。"
    "構図・色・明るさ・追加/削除された要素・雰囲気の違いを具体的に説明してください。"
)


class VisionEngine:
    def __init__(self, chat: ChatEngine):
        self.chat = chat

    def stream(
        self,
        history: list[ContextMessage],
        targets: list[ContextImage],
        params: ChatParams,
        cancel: threading.Event | None = None,
        compare: bool = False,
        content_policy: str = "open",
        system_extra: str | None = None,
    ) -> Iterator[ChatDelta]:
        if not targets:
            raise ValueError("Vision には画像が必要です。画像を添付するか、先に画像を生成してください。")
        history = list(history)
        if compare and len(targets) >= 2 and history:
            last = history[-1]
            hint = COMPARE_HINT.format(n=len(targets), first=targets[0].caption, last=targets[-1].caption)
            history[-1] = ContextMessage(last.role, f"{last.text}\n\n{hint}", last.images)
        yield from self.chat.stream(
            history,
            params,
            cancel,
            extra_images=targets,
            system_extra=system_extra,
            content_policy=content_policy,
        )
