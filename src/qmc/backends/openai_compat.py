"""Minimal streaming client for OpenAI-compatible ``/v1/chat/completions``.

Works with llama.cpp ``llama-server`` (local), vLLM, or any OpenAI-compatible server. Qwen
thinking is toggled with ``chat_template_kwargs.enable_thinking`` (supported by llama-server and
vLLM); thoughts arrive in ``reasoning_content`` when llama-server runs with
``--reasoning-format deepseek``. ``<think>`` tags inside ``content`` are also handled.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator

import requests

from .base import Cancelled, ChatDelta, ChatParams


def parse_sse_lines(lines: Iterable[str | bytes]) -> Iterator[dict]:
    """Yield JSON payloads from Server-Sent-Event lines, stopping at ``[DONE]``."""
    for raw in lines:
        if raw is None:
            continue
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


class ThinkTagSplitter:
    """Split a streamed text into reasoning / content when the server leaves ``<think>`` inline."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.buf = ""

    def feed(self, text: str) -> ChatDelta:
        self.buf += text
        out = ChatDelta()
        while self.buf:
            tag = self.CLOSE if self.in_think else self.OPEN
            idx = self.buf.find(tag)
            if idx >= 0:
                self._emit(out, self.buf[:idx])
                self.buf = self.buf[idx + len(tag) :]
                self.in_think = not self.in_think
                continue
            # keep a possible partial tag at the end of the buffer
            keep = 0
            for k in range(1, len(tag)):
                if self.buf.endswith(tag[:k]):
                    keep = k
            emit, self.buf = (self.buf[:-keep], self.buf[-keep:]) if keep else (self.buf, "")
            self._emit(out, emit)
            break
        return out

    def flush(self) -> ChatDelta:
        out = ChatDelta()
        self._emit(out, self.buf)
        self.buf = ""
        return out

    def _emit(self, out: ChatDelta, text: str) -> None:
        if not text:
            return
        if self.in_think:
            out.reasoning += text
        else:
            out.content += text


def build_payload(model: str, messages: list[dict], params: ChatParams, stream: bool = True) -> dict:
    payload = {"model": model, "messages": messages, "stream": stream, **params.resolved()}
    payload["chat_template_kwargs"] = {"enable_thinking": bool(params.thinking)}
    return payload


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str | None = None, model: str = "local", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def stream_chat(
        self, messages: list[dict], params: ChatParams, cancel: threading.Event | None = None
    ) -> Iterator[ChatDelta]:
        payload = build_payload(self.model, messages, params, stream=True)
        splitter = ThinkTagSplitter()
        with requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=(10, self.timeout),
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"Chat API error {resp.status_code}: {resp.text[:500]}")
            for event in parse_sse_lines(resp.iter_lines()):
                if cancel is not None and cancel.is_set():
                    # closing the response aborts generation on llama-server
                    raise Cancelled()
                if "error" in event:
                    raise RuntimeError(f"Chat API error: {event['error']}")
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    out = splitter.feed(delta.get("content") or "")
                    out.reasoning = (delta.get("reasoning_content") or "") + out.reasoning
                    out.finish_reason = choice.get("finish_reason")
                    if out.content or out.reasoning or out.finish_reason:
                        yield out
        tail = splitter.flush()
        if tail.content or tail.reasoning:
            yield tail

    def complete(self, messages: list[dict], params: ChatParams) -> str:
        """Non-streaming convenience wrapper (content only)."""
        return "".join(d.content for d in self.stream_chat(messages, params))

    def health(self, base_host_url: str | None = None) -> bool:
        url = base_host_url or self.base_url.removesuffix("/v1")
        try:
            return requests.get(f"{url}/health", timeout=3).status_code == 200
        except requests.RequestException:
            return False
