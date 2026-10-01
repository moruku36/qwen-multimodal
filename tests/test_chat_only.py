"""CPU-only checks for the separate Q8 chat path."""

import json
from pathlib import Path

from qmc.app import build_app
from qmc.chat_engine import ContextMessage, build_messages, system_prompt_now
from qmc.config import load_config
from qmc.controller import TurnOptions
from qmc.gpu_manager import NO_GPU


def test_chat_only_has_no_image_model_or_generation_route(tmp_path):
    cfg = load_config(data_dir=tmp_path / "drive", local_db_path=tmp_path / "local.db",
                      mock=True, chat_only=True, web_search="off")
    app = build_app(cfg, gpu=NO_GPU)
    assert set(app.manager.models) == {"chat"}
    assert app.controller.images is None
    sid = app.sessions.create_session()
    events = list(app.controller.handle(sid, "Draw a castle", options=TurnOptions(web_search="off")))
    assert next(event.data.intent.value for event in events if event.kind == "route") == "chat"
    assert app.sessions.session_images(sid) == []
    assert "does not generate or edit" in system_prompt_now(chat_only=True)


def test_chat_only_keeps_image_understanding(tmp_path, make_png):
    cfg = load_config(data_dir=tmp_path / "drive", local_db_path=tmp_path / "local.db",
                      mock=True, chat_only=True, web_search="off")
    app = build_app(cfg, gpu=NO_GPU)
    sid = app.sessions.create_session()
    events = list(app.controller.handle(sid, "What is in this picture?", [str(make_png())],
                                        TurnOptions(web_search="off")))
    assert next(event.data.intent.value for event in events if event.kind == "route") == "vision"
    assert any(event.kind == "text" for event in events)
    assert set(app.manager.models) == {"chat"}


def _text_chars(content) -> int:
    """Characters of text payload in a message content (plain string or list of content parts)."""
    if isinstance(content, str):
        return len(content)
    return sum(len(part["text"]) for part in content if part.get("type") == "text")


def test_long_recent_message_keeps_more_text_with_total_cap():
    text = "A" * 5000 + "END"
    messages = build_messages([ContextMessage("user", text)], max_text_chars=14000)
    assert _text_chars(messages[-1]["content"]) == len(text)
    assert messages[-1]["content"] == text
    many = [ContextMessage("user", str(n) * 8000) for n in range(10)]
    bounded = build_messages(many, max_text_chars=14000)
    # Consecutive user messages are merged into content parts, so count only the text payloads.
    assert sum(_text_chars(item["content"]) for item in bounded[1:]) <= 14000


def test_chat_notebook_only_prefetches_chat():
    notebook = json.loads(Path("Qwen-Q8-Chat-Colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "prefetch_models(chat=True, image=False)" in source
    assert "chat_only=True" in source
    assert "requirements-chat-colab.txt" in source
    assert "show_shutdown_button()" in source
    assert "QMC_IMAGE_" not in source
