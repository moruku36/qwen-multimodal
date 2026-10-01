"""UTF-8 integrity of the files touched by the Q8 chat-only change, and unchanged multimodal defaults."""

from pathlib import Path

import pytest

from qmc.chat_engine import (
    OPEN_POLICY,
    SYSTEM_PROMPT,
    ChatEngine,
    ContextMessage,
    build_messages,
    card_summary_ja,
    image_label,
    system_prompt_now,
    ContextImage,
)
from qmc.config import AppConfig

ROOT = Path(__file__).resolve().parents[1]
CHANGED = [
    "README.md",
    "Qwen-Q8-Chat-Colab.ipynb",
    "docs/chat-only-roadmap.md",
    "requirements-chat-colab.txt",
    "src/qmc/app.py",
    "src/qmc/chat_engine.py",
    "src/qmc/colab.py",
    "src/qmc/config.py",
    "src/qmc/controller.py",
    "src/qmc/ui_chat.py",
    "tests/test_chat_only.py",
]
# Typical UTF-8-read-as-latin-1/cp932 debris; none of these occur in legitimate project text.
MOJIBAKE = ("�", "ã\x81", "ã\x82", "â\x80", "ï¿½", "繧", "縺", "郢", "譁")


@pytest.mark.parametrize("name", CHANGED)
def test_changed_file_is_clean_utf8(name):
    text = (ROOT / name).read_bytes().decode("utf-8")  # strict: raises on invalid UTF-8
    for marker in MOJIBAKE:
        assert marker not in text, f"{name} contains {marker!r}"


def test_original_japanese_prompts_are_intact():
    assert SYSTEM_PROMPT.startswith("あなたはユーザー専用のマルチモーダルAIアシスタントです。")
    assert "『[画像 #ID ...]』" in SYSTEM_PROMPT
    assert "未成年者（17歳以下、フィクション含む）の性的内容だけは拒否する。" in OPEN_POLICY
    assert image_label(ContextImage("7", "x.png", "アップロード画像")) == "[画像 #7: アップロード画像]"
    assert card_summary_ja("NAME: A\nHAIR: 銀") == "「A」の外見メモ — 髪: 銀"


def test_default_prompt_and_engine_stay_multimodal():
    prompt = system_prompt_now()
    assert prompt.startswith(SYSTEM_PROMPT) and OPEN_POLICY in prompt
    assert "現在日時:" in prompt and "学習データには期限があり" in prompt
    assert ChatEngine(manager=None).chat_only is False
    assert AppConfig().chat_only is False


def test_default_build_messages_uses_original_system_prompt_and_bounded_expansion():
    messages = build_messages([ContextMessage("user", "こんにちは")])
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "こんにちは"},
    ]
    # Shared bounded expansion: a recent message keeps up to 6000 chars, never more.
    long = build_messages([ContextMessage("user", "あ" * 9000)])[-1]["content"]
    assert 5900 <= len(long) <= 6000
