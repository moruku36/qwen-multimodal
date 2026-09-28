import pytest

from qmc.backends.base import ChatParams
from qmc.backends.mock import MockChatModel
from qmc.chat_engine import ChatEngine, ContextImage, ContextMessage
from qmc.gpu_manager import PROFILES
from qmc.model_manager import ModelManager
from qmc.vision_engine import VisionEngine


def _engine():
    mm = ModelManager(profile=PROFILES["cpu"])
    chat = MockChatModel()
    mm.register(chat)
    return VisionEngine(ChatEngine(mm)), chat


def test_vision_sends_target_images(make_png):
    engine, chat = _engine()
    img = ContextImage("u1", str(make_png()), "アップロード画像")
    out = "".join(
        d.content for d in engine.stream([ContextMessage("user", "何が写ってる？")], [img], ChatParams())
    )
    assert "1 枚" in out
    last = chat.last_messages[-1]["content"]
    assert any(p["type"] == "image_url" for p in last)


def test_compare_adds_hint_and_both_images(make_png):
    engine, chat = _engine()
    a = ContextImage("a", str(make_png("a.png")), "元画像")
    b = ContextImage("b", str(make_png("b.png")), "現在の画像")
    list(engine.stream([ContextMessage("user", "違いは？")], [a, b], ChatParams(), compare=True))
    last = chat.last_messages[-1]["content"]
    assert sum(p["type"] == "image_url" for p in last) == 2
    assert "比較" in str(last)


def test_vision_without_images_errors():
    engine, _ = _engine()
    with pytest.raises(ValueError):
        list(engine.stream([ContextMessage("user", "x")], [], ChatParams()))
