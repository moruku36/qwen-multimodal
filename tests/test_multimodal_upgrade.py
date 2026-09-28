import hashlib
from types import SimpleNamespace

import pytest
from PIL import Image

from qmc.app import build_app
from qmc.config import load_config
from qmc.controller import TurnOptions
from qmc.gpu_manager import NO_GPU
from qmc.image_engine import ImageOptions
from qmc.imaging import render_pdf_pages, slice_tall_image
from qmc.router import ImageTarget, Intent, RouteContext, route
from qmc.ui import gallery_items


@pytest.fixture
def app(tmp_path):
    cfg = load_config(data_dir=tmp_path / "drive", local_db_path=tmp_path / "local.db", mock=True)
    result = build_app(cfg, gpu=NO_GPU)
    for model in result.manager.models.values():
        model.load_delay = 0
        if hasattr(model, "token_delay"):
            model.token_delay = 0
        if hasattr(model, "step_delay"):
            model.step_delay = 0
    return result


def run(app, sid, text, files=None, options=None):
    return list(app.controller.handle(sid, text, files, options or TurnOptions(web_search="off")))


@pytest.mark.parametrize(
    ("text", "intent", "target", "n"),
    [
        ("元に戻して", Intent.RESTORE, ImageTarget.LATEST, 0),
        ("オリジナルに戻して", Intent.RESTORE, ImageTarget.LATEST, 0),
        ("revert", Intent.RESTORE, ImageTarget.LATEST, 0),
        ("元に戻して明るくして", Intent.EDIT, ImageTarget.ROOT, 0),
        ("2個前を編集", Intent.EDIT, ImageTarget.NTH, 2),
        ("ふたつ前", Intent.VISION, ImageTarget.NTH, 2),
        ("前の画像", Intent.VISION, ImageTarget.NTH, 1),
    ],
)
def test_lineage_router(text, intent, target, n):
    decision = route(text, RouteContext(has_session_image=True, turns_since_last_image=0))
    assert (decision.intent, decision.target, decision.n) == (intent, target, n)


def test_selected_image_routes_to_edit_even_after_many_turns():
    decision = route(
        "この画像を編集",
        RouteContext(has_session_image=True, has_selected_image=True, turns_since_last_image=10),
    )
    assert (decision.intent, decision.target) == (Intent.EDIT, ImageTarget.SELECTED)


def test_restore_copies_root_without_image_backend(app):
    sid = app.sessions.create_session()
    run(app, sid, "猫の絵を描いて")
    root = app.sessions.latest_image(sid)
    run(app, sid, "もう少し明るくして")
    edited = app.sessions.latest_image(sid)
    count = len(app.manager.get("image").requests)
    run(app, sid, "元に戻して")
    restored = app.sessions.latest_image(sid)
    assert [r.id for r in app.sessions.lineage(restored.id)] == [root.id, edited.id, restored.id]
    assert restored.parent_id == edited.id and restored.meta["restored_from"] == root.id

    def digest(image):
        return hashlib.sha256(app.sessions.image_path(image).read_bytes()).hexdigest()

    assert digest(restored) == digest(root)
    assert len(app.manager.get("image").requests) == count


def test_gallery_items_root_to_leaf():
    lineage = [SimpleNamespace(id="a", revision=0), SimpleNamespace(id="b", revision=1)]
    assert gallery_items(lineage, lambda image: f"/{image.id}.png") == [
        ("/a.png", "rev0 元"),
        ("/b.png", "rev1"),
    ]


def test_three_uploaded_references_are_used_in_order(app, make_png):
    sid = app.sessions.create_session()
    paths = [str(make_png(name=f"{i}.png", size=(70 + i, 50 + i))) for i in range(3)]
    events = run(app, sid, "顔は1枚目、背景は2枚目で変えて", paths)
    assert next(e.data for e in events if e.kind == "route").intent == Intent.EDIT
    request = app.manager.get("image").requests[-1]
    assert [image.size for image in request.images] == [(70, 50), (71, 51), (72, 52)]
    assert len(app.sessions.generations(sid)[-1]["source_image_ids"]) == 3


def test_uploads_plus_selected_image_puts_selection_last(app, make_png):
    sid = app.sessions.create_session()
    run(app, sid, "猫の絵を描いて")
    selected = app.sessions.latest_image(sid)
    paths = [str(make_png(name=f"ref{i}.png", size=(70 + i, 50 + i))) for i in range(2)]
    run(
        app,
        sid,
        "この顔と服装で背景を変えて",
        paths,
        TurnOptions(web_search="off", selected_image_id=selected.id, image=ImageOptions(steps=1)),
    )
    sources = app.sessions.generations(sid)[-1]["source_image_ids"]
    assert len(sources) == 3 and sources[-1] == selected.id
    assert app.sessions.latest_image(sid).parent_id == selected.id


def test_reference_images_are_capped_at_ten(app, make_png):
    sid = app.sessions.create_session()
    paths = [str(make_png(name=f"many{i}.png")) for i in range(11)]
    run(app, sid, "背景を変えて", paths)
    assert len(app.manager.get("image").requests[-1].images) == 10


def test_pdf_pages_reach_vision(app, tmp_path):
    pytest.importorskip("pypdfium2")
    path = tmp_path / "document.pdf"
    Image.new("RGB", (80, 120), "red").save(
        path, "PDF", save_all=True, append_images=[Image.new("RGB", (80, 120), "blue")]
    )
    pages = render_pdf_pages(path)
    assert len(pages) == 2
    sid = app.sessions.create_session()
    run(app, sid, "要約して", [str(path)])
    message = app.sessions.get_messages(sid)[0]
    assert [image.meta["pdf_page"] for image in message.images] == [1, 2]
    content = app.manager.get("chat").last_messages[-1]["content"]
    assert sum(part.get("type") == "image_url" for part in content) == 2


def test_pdf_only_uses_summary_prompt(app, tmp_path):
    pytest.importorskip("pypdfium2")
    path = tmp_path / "single.pdf"
    Image.new("RGB", (80, 120)).save(path, "PDF")
    sid = app.sessions.create_session()
    run(app, sid, "", [str(path)])
    assert app.sessions.get_messages(sid)[0].content.startswith("この資料の内容をページ順に要約")


def test_tall_screenshot_slices_reach_vision(app, make_png):
    path = make_png(size=(400, 2000))
    assert len(slice_tall_image(path, max_side=512)) >= 4
    app.controller.max_image_side = 512
    sid = app.sessions.create_session()
    run(app, sid, "何が写っていますか？", [str(path)])
    uploaded = app.sessions.get_messages(sid)[0].images[0]
    assert uploaded.meta["sliced"] and uploaded.meta["bands"] >= 4
    content = app.manager.get("chat").last_messages[-1]["content"]
    assert sum(part.get("type") == "image_url" for part in content) == uploaded.meta["bands"]


def test_four_variations_share_one_message_and_use_distinct_seeds(app):
    sid = app.sessions.create_session()
    run(
        app,
        sid,
        "猫の絵を描いて",
        options=TurnOptions(web_search="off", image=ImageOptions(steps=1, seed=123, variations=4)),
    )
    assistant = app.sessions.get_messages(sid)[-1]
    assert len(assistant.images) == 4
    assert [image.seed for image in assistant.images] == [123, 8042, 15961, 23880]
    assert all(image.meta["variation_of"] == assistant.images[0].id for image in assistant.images)
    assert len(app.manager.get("image").requests) == 4


def test_cancel_after_first_variation_keeps_first(app, monkeypatch):
    original = app.controller.images.run

    def stop_after_first(*args, **kwargs):
        result = original(*args, **kwargs)
        app.controller.cancel()
        return result

    monkeypatch.setattr(app.controller.images, "run", stop_after_first)
    sid = app.sessions.create_session()
    run(
        app,
        sid,
        "猫の絵を描いて",
        options=TurnOptions(web_search="off", image=ImageOptions(steps=1, variations=4)),
    )
    assert len(app.sessions.get_messages(sid)[-1].images) == 1


def test_mock_audio_is_saved_as_user_text(app, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"mock")
    sid = app.sessions.create_session()
    run(app, sid, "", [str(audio)])
    assert app.sessions.get_messages(sid)[0].content == "（音声の文字起こしモック）"
    sid2 = app.sessions.create_session()
    run(app, sid2, "続けて答えて", [str(audio)])
    assert app.sessions.get_messages(sid2)[0].content == "（音声の文字起こしモック）\n続けて答えて"
