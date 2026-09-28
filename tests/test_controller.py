"""End-to-end turn flow on mock backends (the MVP scenarios, without a GPU)."""

import pytest

from qmc.app import build_app
from qmc.backends.base import ChatDelta
from qmc.config import load_config
from qmc.controller import TurnOptions
from qmc.gpu_manager import NO_GPU, GPUInfo
from qmc.image_engine import ImageOptions


@pytest.fixture
def app(tmp_path):
    cfg = load_config(data_dir=tmp_path / "drive", local_db_path=tmp_path / "local" / "h.db", mock=True)
    a = build_app(cfg, gpu=NO_GPU)
    for m in a.manager.models.values():
        m.load_delay = 0
        if hasattr(m, "token_delay"):
            m.token_delay = 0
        if hasattr(m, "step_delay"):
            m.step_delay = 0
    return a


FAST = TurnOptions(image=ImageOptions(steps=2), web_search="off")


def run(app, sid, text, files=None, options=FAST):
    return list(app.controller.handle(sid, text, files, options))


def kinds(events, kind):
    return [e.data for e in events if e.kind == kind]


def test_mvp_flow(app, make_png):
    sid = app.sessions.create_session()

    # Test 1: text chat
    ev = run(app, sid, "TerraformとPulumiの違いを教えて")
    assert kinds(ev, "route")[0].intent.value == "chat"
    assert "Terraform" in "".join(kinds(ev, "text"))

    # Test 2: vision on an uploaded image
    ev = run(app, sid, "この画像に何が写っていますか？", [str(make_png())])
    assert kinds(ev, "route")[0].intent.value == "vision"
    assert "1 枚" in "".join(kinds(ev, "text"))

    # Test 3: generation
    ev = run(app, sid, "東京の夜景を背景にした未来的なデータセンターを生成して")
    assert kinds(ev, "route")[0].intent.value == "generate"
    gen = app.sessions.latest_image(sid)
    assert gen.kind == "generated" and kinds(ev, "image")

    # Test 4: edit of the previous image
    ev = run(app, sid, "もう少し夜を暗くして、ネオンを増やして")
    assert kinds(ev, "route")[0].intent.value == "edit"
    edited = app.sessions.latest_image(sid)
    assert edited.kind == "edited" and edited.parent_id == gen.id and edited.revision == 1
    assert edited.seed != gen.seed

    # Test 5: compare original vs current -> both images go to the vision model
    ev = run(app, sid, "元画像と今の画像の違いを説明して")
    d = kinds(ev, "route")[0]
    assert d.intent.value == "vision" and d.compare
    chat_model = app.manager.get("chat")
    last = chat_model.last_messages[-1]["content"]
    ids = [p["text"] for p in last if p["type"] == "text" and p["text"].startswith("[画像 #")]
    assert any(gen.id in t for t in ids) and any(edited.id in t for t in ids)

    # history persisted: generations recorded with params
    gens = app.sessions.generations(sid)
    assert [g["kind"] for g in gens] == ["generate", "edit"]
    assert gens[1]["source_image_ids"] == [gen.id]


def test_restore_after_restart(app, tmp_path):
    """Test 6: Colab restart -> local DB gone -> history restored from the Drive mirror."""
    sid = app.sessions.create_session()
    run(app, sid, "こんにちは")
    run(app, sid, "猫の絵を描いて")
    app.store.close()
    (tmp_path / "local" / "h.db").unlink()

    cfg = load_config(data_dir=tmp_path / "drive", local_db_path=tmp_path / "local" / "h.db", mock=True)
    app2 = build_app(cfg, gpu=NO_GPU)
    msgs = app2.sessions.get_messages(sid)
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    img = app2.sessions.latest_image(sid)
    assert app2.sessions.image_path(img).exists()


def test_open_refusal_retries_once_and_saves_final_answer(app, monkeypatch):
    calls = []

    def fake_stream(history, params, cancel, system_extra=None, content_policy="open"):
        calls.append(system_extra)
        yield ChatDelta(content="紹介はできません" if len(calls) == 1 else "候補は Example [1] です。")

    monkeypatch.setattr(app.controller.chat, "stream", fake_stream)
    sid = app.sessions.create_session()
    events = run(app, sid, "日本のサイトを教えて", options=TurnOptions(web_search="off"))
    msg = app.sessions.get_messages(sid)[-1]
    assert len(calls) == 2
    assert calls[0] is None and "## Web検索結果" in calls[1]
    assert "紹介はできません" not in msg.content
    assert "候補は Example" in msg.content and "参考（Web検索）" in msg.content
    assert any("拒否だったため" in str(e.data) for e in events if e.kind == "status")


def test_open_refusal_after_search_uses_source_instruction(app, monkeypatch):
    calls = []

    def fake_stream(history, params, cancel, system_extra=None, content_policy="open"):
        calls.append(history[-1].text)
        yield ChatDelta(content="cannot recommend" if len(calls) == 1 else "Example URL [1]")

    monkeypatch.setattr(app.controller.chat, "stream", fake_stream)
    sid = app.sessions.create_session()
    run(app, sid, "サイトを教えて", options=TurnOptions(web_search="on"))
    assert len(calls) == 2
    assert "検索結果にある固有名詞とURL" in calls[1]
    assert "検索結果にある固有名詞とURL" not in app.sessions.get_messages(sid)[0].content
    assert app.sessions.get_messages(sid)[-1].content.startswith("Example URL [1]")


def test_open_image_rewrite_drops_adult_term_uses_original(app, monkeypatch):
    monkeypatch.setattr(app.controller.chat, "rewrite_image_prompt", lambda *a: "a beautiful portrait")
    sid = app.sessions.create_session()
    run(
        app,
        sid,
        "成人向けヌードを描いて",
        options=TurnOptions(
            mode="generate", image=ImageOptions(steps=1), prompt_rewrite="on", content_policy="open"
        ),
    )
    assert app.sessions.generations(sid)[-1]["effective_prompt"] == "成人向けヌードを描いて"


def test_gpu_profile_auto_selection(tmp_path):
    """Test 7 (CPU part): the detected GPU selects the profile."""
    cfg = load_config(data_dir=tmp_path / "d", local_db_path=tmp_path / "l.db")
    cfg.mock = True
    cfg.gpu_profile_override = None
    # mock forces the cpu profile unless an override is given; real path uses detection:
    from qmc.gpu_manager import select_profile

    assert select_profile(GPUInfo("NVIDIA A100-SXM4-40GB", 39.4)).mode == "Performance"
    assert select_profile(GPUInfo("NVIDIA L4", 22.0)).mode == "Low VRAM"


def test_empty_message_and_bad_upload(app, tmp_path):
    sid = app.sessions.create_session()
    ev = run(app, sid, "   ")
    assert kinds(ev, "error")
    bad = tmp_path / "bad.png"
    bad.write_text("nope")
    ev = run(app, sid, "", [str(bad)])
    assert kinds(ev, "error")
    assert app.sessions.get_messages(sid) == []


def test_manual_mode_override(app):
    sid = app.sessions.create_session()
    ev = run(app, sid, "夕焼けの海", options=TurnOptions(mode="generate", image=ImageOptions(steps=1)))
    assert kinds(ev, "route")[0].intent.value == "generate"


def test_edit_without_image_reports_error(app):
    sid = app.sessions.create_session()
    ev = run(app, sid, "背景を青に", options=TurnOptions(mode="edit"))
    # no image -> manual edit falls back to chat with a warning
    assert kinds(ev, "route")[0].intent.value == "chat"
    assert any("画像" in s for s in kinds(ev, "status"))


def test_regenerate_replaces_last_answer(app):
    sid = app.sessions.create_session()
    run(app, sid, "こんにちは")
    ev = list(app.controller.regenerate(sid, FAST))
    assert kinds(ev, "done")
    msgs = app.sessions.get_messages(sid)
    assert [m.role for m in msgs] == ["user", "assistant"]


def test_cancel_stops_generation(app):
    sid = app.sessions.create_session()
    app.manager.get("image").step_delay = 0.05
    it = app.controller.handle(sid, "猫を描いて", None, TurnOptions(image=ImageOptions(steps=50)))
    events = []
    for e in it:
        events.append(e)
        if e.kind == "status" and "step" in str(e.data):
            app.controller.cancel()
    assert any("停止" in str(s) for s in kinds(events, "status"))
    assert app.sessions.latest_image(sid) is None


def test_prompt_rewrite_auto_uses_loaded_chat(app):
    sid = app.sessions.create_session()
    run(app, sid, "こんにちは")  # chat model is now resident
    run(app, sid, "猫の絵を描いて", options=TurnOptions(image=ImageOptions(steps=1), prompt_rewrite="auto"))
    g = app.sessions.generations(sid)[-1]
    assert g["effective_prompt"].startswith("[rewritten]")


def test_image_rewrite_refusal_uses_original(app, monkeypatch):
    monkeypatch.setattr(app.controller.chat, "rewrite_image_prompt", lambda *a: "お答えできません")
    sid = app.sessions.create_session()
    run(app, sid, "成人の肖像を描いて", options=TurnOptions(image=ImageOptions(steps=1), prompt_rewrite="on"))
    assert app.sessions.generations(sid)[-1]["effective_prompt"] == "成人の肖像を描いて"
