"""Lazy CPU-friendly speech synthesis for the visible assistant answer."""

from __future__ import annotations

import asyncio
import re
import wave
from pathlib import Path
from typing import Protocol


def speakable_text(text: str) -> str:
    main = text.split("\n\n**🔎 参考（Web検索）**", 1)[0]
    main = re.split(r"\n(?:#{1,3}\s*)?(?:Sources|出典|参照元)\s*\n", main, maxsplit=1, flags=re.I)[0]
    main = re.sub(r"<[^>]*>", "", main)
    main = re.sub(r"https?://\S+", "", main)
    main = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", main)
    return main.strip()[:4000]


class TTSBackend(Protocol):
    def synthesize(self, text: str, path: Path) -> Path: ...


class EdgeTTS:
    def __init__(self, voice: str = "ja-JP-NanamiNeural"):
        self.voice = voice

    def synthesize(self, text: str, path: Path) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("読み上げのパッケージがありません（edge-tts）") from exc

        async def save() -> None:
            await edge_tts.Communicate(text, self.voice).save(str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(save())
        return path


class MockTTS:
    def __init__(self):
        self.calls: list[str] = []

    def synthesize(self, text: str, path: Path) -> Path:
        self.calls.append(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 1600)
        return path
