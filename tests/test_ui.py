import gradio as gr
import pytest

from qmc.app import build_app
from qmc.config import load_config
from qmc.controller import TurnOptions
from qmc.gpu_manager import NO_GPU
from qmc.image_engine import ImageOptions
from qmc.ui import build_ui, pending_messages, render_history, status_markdown


@pytest.fixture
def app(tmp_path):
    cfg = load_config(data_dir=tmp_path / "drive", local_db_path=tmp_path / "l" / "h.db", mock=True)
    return build_app(cfg, gpu=NO_GPU)


def test_build_ui(app):
    assert isinstance(build_ui(app), gr.Blocks)


def test_render_history_with_images_and_reasoning(app):
    sid = app.sessions.create_session()
    list(app.controller.handle(sid, "猫を描いて", None, TurnOptions(image=ImageOptions(steps=1))))
    list(app.controller.handle(sid, "こんにちは", None, TurnOptions(thinking=True)))
    msgs = render_history(app, sid)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user" and "assistant" in roles
    assert any(
        isinstance(c, dict) and c.get("path", "").endswith(".png")
        for m in msgs
        for c in m["content"]
        if isinstance(m["content"], list)
    )
    assert any(m.get("metadata", {}).get("title", "").startswith("🧠") for m in msgs)


def test_render_unknown_session(app):
    assert render_history(app, None) == []
    assert render_history(app, "missing") == []


def test_pending_and_status(app):
    p = pending_messages("r", "", "loading", ["bad"])
    assert p[0]["metadata"]["status"] == "pending"
    assert "loading" in p[1]["content"] and "bad" in p[1]["content"]
    md = status_markdown(app)
    assert "CPU (mock)" in md and "Loaded" in md
