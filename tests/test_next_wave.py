import base64
import io
import shutil
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw
from scripts.image_http_stub import make_server

from qmc.app import build_app
from qmc.backends.base import ImageRequest
from qmc.backends.image_http import HttpImageModel
from qmc.chat_engine import ContextMessage, build_messages
from qmc.config import load_config
from qmc.controller import TurnOptions
from qmc.gpu_manager import NO_GPU
from qmc.image_engine import ImageOptions
from qmc.imaging import mask_from_editor, video_sample_times
from qmc.router import ImageTarget, Intent, RouteContext, route
from qmc.tts import speakable_text
from qmc.ui import build_ui, lineage_view


@pytest.fixture
def app(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data", local_db_path=tmp_path / "local.db", mock=True)
    cfg.web_search = "off"
    result = build_app(cfg, gpu=NO_GPU)
    for model in result.manager.models.values():
        model.load_delay = 0
        if hasattr(model, "token_delay"):
            model.token_delay = 0
        if hasattr(model, "step_delay"):
            model.step_delay = 0
    return result


def send(app, sid, text, files=None, **kwargs):
    return list(
        app.controller.handle(
            sid, text, files, TurnOptions(web_search="off", image=ImageOptions(steps=1, **kwargs))
        )
    )


def test_long_history_is_budgeted_and_still_keeps_latest():
    history = [ContextMessage("user", f"turn{i} " + "x" * 4000) for i in range(50)]
    messages = build_messages(history, max_text_chars=12000)
    body = " ".join(str(message["content"]) for message in messages)
    assert len(body) < 14000
    assert "turn49" in body
    assert "turn0" not in body


def test_context_error_retries_without_deleting_session(app):
    sid = app.sessions.create_session()
    for index in range(4):
        send(app, sid, f"質問{index}")
    model = app.manager.get("chat")
    original = model.stream_chat
    calls = []

    def fail_once(messages, params, cancel=None):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("Chat API error 400: context length exceeded")
        yield from original(messages, params, cancel)

    model.stream_chat = fail_once
    events = send(app, sid, "最後の質問")
    assert not [event for event in events if event.kind == "error"]
    assert len(calls) == 2
    assert len(app.sessions.get_messages(sid)) == 10


def test_four_variations_show_siblings_and_selected_edit(app):
    sid = app.sessions.create_session()
    send(app, sid, "猫を描いて", variations=4)
    gallery, ids, selected, _ = lineage_view(app, sid)
    assert [label for _, label in gallery] == ["var1", "var2", "var3", "var4"]
    assert selected == ids[0]
    assert build_ui(app) is not None
    decision = route("3枚目を少し暗く", RouteContext(has_session_image=True, has_selected_image=True))
    assert (decision.intent, decision.target, decision.n) == (Intent.EDIT, ImageTarget.VARIATION, 3)
    send(app, sid, "3枚目を少し暗く")
    assert app.sessions.latest_image(sid).parent_id == ids[2]


def test_mask_editor_produces_full_size_binary_mask_and_edit_persists(app):
    source = Image.new("RGB", (120, 80), "white")
    layer = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    ImageDraw.Draw(layer).line((15, 10, 80, 60), fill=(255, 0, 0, 255), width=2)
    mask = mask_from_editor({"layers": [layer]}, source.size)
    assert mask.size == source.size and mask.getbbox()
    assert set(mask.getdata()) <= {0, 255}
    sid = app.sessions.create_session()
    send(app, sid, "猫を描いて")
    selected = app.sessions.latest_image(sid)
    options = TurnOptions(
        web_search="off", image=ImageOptions(steps=1), selected_image_id=selected.id, mask_image=mask
    )
    list(app.controller.handle(sid, "この画像の背景を変えて", options=options))
    edited = app.sessions.latest_image(sid)
    assert edited.parent_id == selected.id
    assert edited.meta.get("mask_path")
    assert app.manager.get("image").requests[-1].mask_image is not None


def test_video_times_include_ends():
    times = video_sample_times(10.0)
    assert len(times) == 8 and times[0] == 0 and times[-1] == pytest.approx(9.9)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_video_has_one_poster_and_multiple_vision_frames(app, tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=1",
            "-t",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    sid = app.sessions.create_session()
    events = send(app, sid, "何が映ってる？", [str(video)])
    assert not [event for event in events if event.kind == "error"]
    user = app.sessions.get_messages(sid)[0]
    assert len(user.images) == 1 and user.images[0].meta["video"]
    assert user.images[0].meta["frames"] >= 2


def test_mock_tts_called_once_without_sources(app):
    sid = app.sessions.create_session()
    send(app, sid, "こんにちは")
    last = app.sessions.get_messages(sid)[-1]
    app.sessions.update_message(last.id, content="回答です。\n出典\nhttps://example.com")
    audio = app.controller.read_last_answer(sid)
    assert audio and audio.endswith(".wav")
    assert app.controller.tts.calls == ["回答です。"]
    assert (
        speakable_text("回答です。\n\n**🔎 参考（Web検索）**\n1. [source](https://example.com)")
        == "回答です。"
    )


def test_http_image_backend_generate_and_edit_with_mask():
    image = Image.new("RGB", (16, 16), "purple")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    payload = {"data": [{"b64_json": base64.b64encode(stream.getvalue()).decode()}]}
    response = SimpleNamespace(status_code=200, json=lambda: payload, raise_for_status=lambda: None)
    model = HttpImageModel("https://image.example", "test")
    with patch("qmc.backends.image_http.requests.post", return_value=response) as post:
        assert model.generate(ImageRequest(prompt="cat")).size == (16, 16)
        assert post.call_args.args[0].endswith("/v1/images/generations")
        model.generate(ImageRequest(prompt="edit", images=[image], mask_image=Image.new("L", (16, 16), 255)))
        assert post.call_args.args[0].endswith("/v1/images/edits")
        assert any(part[0] == "mask" for part in post.call_args.kwargs["files"])


def test_remote_image_app_does_not_import_diffusers(tmp_path):
    cfg = load_config(data_dir=tmp_path / "data", local_db_path=tmp_path / "local.db", mock=False)
    cfg.web_search = "off"
    cfg.image.remote_base_url = "https://image.example"
    app = build_app(cfg, gpu=NO_GPU)
    assert isinstance(app.manager.get("image"), HttpImageModel)
    assert not app.manager.get("image").uses_local_gpu
    cfg.mock = True
    cfg.local_db_path = tmp_path / "mock-local.db"
    assert isinstance(build_app(cfg, gpu=NO_GPU).manager.get("image"), HttpImageModel)
    cfg.image.remote_api_key = "secret"
    assert "secret" not in str(cfg.public_dict())


def test_http_image_stub_round_trip():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = HttpImageModel(f"http://127.0.0.1:{server.server_port}")
        assert model.generate(ImageRequest(prompt="cat", steps=1)).size == (512, 512)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
