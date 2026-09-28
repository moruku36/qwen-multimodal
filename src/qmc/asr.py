"""Lazy speech recognition for uploaded or microphone audio."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ASRBackend(Protocol):
    def transcribe(self, path: str | Path) -> str: ...


class MockASR:
    def transcribe(self, path: str | Path) -> str:
        return "（音声の文字起こしモック）"


class WhisperASR:
    def __init__(self, model: str = "small", device: str = "cpu"):
        if device not in {"cpu", "cuda"}:
            raise ValueError("QMC_ASR_DEVICE は cpu または cuda を指定してください")
        self.model_name = model
        self.device = device
        self._model = None

    def transcribe(self, path: str | Path) -> str:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("音声認識のパッケージがありません（faster-whisper）") from exc
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16",
            )
        segments, _ = self._model.transcribe(str(path), language="ja")
        return " ".join(segment.text.strip() for segment in segments).strip()
