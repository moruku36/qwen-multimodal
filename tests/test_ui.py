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


def test_composer_first_layout_and_mic_wiring(app):
    config = build_ui(app).get_config_file()
    by_name = {c["props"].get("elem_id"): c for c in config["components"]}
    assert by_name["chat-panel"]["props"]["height"] == "calc(100vh - 210px)"
    assert by_name["mic-panel"]["props"]["visible"] is False
    assert by_name["mic-recording"]["props"]["visible"] is False
    assert by_name["voice-submit"]["props"]["visible"] is False
    assert by_name["mic-toggle"]["props"]["visible"] is True
    assert "settings-shell" not in by_name
    mic_click = next(
        dep for dep in config["dependencies"] if (by_name["mic-toggle"]["id"], "click") in dep["targets"]
    )
    assert by_name["mic-recording"]["id"] in mic_click["outputs"]
    assert by_name["mic-panel"]["id"] in mic_click["outputs"]
    voice_click = next(
        dep for dep in config["dependencies"] if (by_name["voice-submit"]["id"], "click") in dep["targets"]
    )
    assert by_name["mic-recording"]["id"] in voice_click["inputs"]
    accordions = {c["props"].get("label"): c for c in config["components"] if c["type"] == "accordion"}
    assert {
        "ツール",
        "画像の系譜・バリエーション",
        "マスクで編集",
        "画像生成・編集の設定",
        "検索・モード",
        "システム状態",
    } <= accordions.keys()
    assert accordions["マスクで編集"]["props"]["visible"] is False


def test_open_policy_label():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src" / "qmc" / "ui.py").read_text(encoding="utf-8")
    assert '[("開放", "open"), ("標準", "standard")]' in source
    assert "検索OK" not in source


def test_render_history_with_images_and_reasoning(app):
    sid = app.sessions.create_session()
    list(app.controller.handle(sid, "猫を描いて", None, TurnOptions(image=ImageOptions(steps=1))))
    list(app.controller.handle(sid, "こんにちは", None, TurnOptions(thinking=True, web_search="off")))
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


def test_image_controls_use_configured_defaults_and_presets(app):
    app.controller.images.default_band = 768
    app.controller.images.default_steps = 32
    config = build_ui(app).get_config_file()
    by_label = {c["props"].get("label"): c for c in config["components"]}
    assert by_label["解像度帯"]["props"]["value"] == 768
    assert by_label["Steps"]["props"]["value"] == 32
    preset = by_label["画像プリセット（GPU・設定上限内 / 適用時は1枚）"]
    assert preset["props"]["value"] == "configured"
    dep = next(d for d in config["dependencies"] if (preset["id"], "change") in d["targets"])
    assert dep["outputs"] == [
        by_label[label]["id"] for label in ("解像度帯", "Steps", "バリエーション（生成のみ）")
    ]
