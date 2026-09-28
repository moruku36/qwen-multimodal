"""Backend-neutral interfaces.

Engines and the UI depend only on these types, never on llama.cpp / diffusers directly, so a
backend can be swapped (Colab local GPU -> RunPod / vLLM / OpenAI-compatible API) without
touching the UI.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image


@dataclass
class ChatDelta:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None


@dataclass
class ChatParams:
    thinking: bool = False
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None

    def resolved(self) -> dict[str, Any]:
        """Qwen3.8 recommended sampling (HF model card) unless explicitly overridden."""
        if self.thinking:
            base = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
        else:
            base = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
        for key in base:
            value = getattr(self, key)
            if value is not None:
                base[key] = value
        base["max_tokens"] = self.max_tokens
        return base


class ChatBackend(Protocol):
    def stream_chat(
        self, messages: list[dict], params: ChatParams, cancel: threading.Event | None = None
    ) -> Iterator[ChatDelta]: ...


@dataclass
class ImageRequest:
    prompt: str
    images: list[Image.Image] = field(default_factory=list)  # condition images (edit) - empty for T2I
    mask_image: Image.Image | None = None  # white pixels are the requested edit region
    width: int | None = None
    height: int | None = None
    output_resolution: int = 1024
    steps: int = 40
    seed: int = 0
    negative_prompt: str | None = None
    true_cfg_scale: float = 1.0

    @property
    def is_edit(self) -> bool:
        return bool(self.images)

    def params_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "output_resolution": self.output_resolution,
            "steps": self.steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "true_cfg_scale": self.true_cfg_scale,
            "num_condition_images": len(self.images),
        }


ProgressFn = Callable[[int, int], None]


class ImageBackend(Protocol):
    model_label: str

    def generate(
        self, request: ImageRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> Image.Image: ...


class Cancelled(Exception):
    """Raised when the user pressed Stop."""
