import threading

import pytest

from qmc.backends.base import Cancelled
from qmc.backends.mock import MockChatModel, MockImageModel
from qmc.gpu_manager import PROFILES
from qmc.image_engine import ImageEngine, ImageOptions, clamp_band, size_for
from qmc.model_manager import ModelManager


def test_size_for_matches_official_ratios():
    assert size_for("1:1", 2048) == (2048, 2048)
    assert size_for("16:9", 2048) == (2752, 1536)
    w, h = size_for("16:9", 1024)
    assert (w % 32, h % 32) == (0, 0)
    assert w > h
    assert size_for("unknown", 1024) == (1024, 1024)


def test_clamp_band_respects_profile():
    assert clamp_band(2048, PROFILES["l4"]) == 1280
    assert clamp_band(2048, PROFILES["a100_80"]) == 2048
    assert clamp_band(100, PROFILES["l4"]) == 512


def _engine(profile="l4"):
    mm = ModelManager(profile=PROFILES[profile])
    chat, image = MockChatModel(), MockImageModel()
    mm.register(chat)
    mm.register(image)
    return ImageEngine(mm, PROFILES[profile]), mm, chat, image


def test_generation_request_uses_profile_defaults():
    engine, *_ = _engine("l4")
    req = engine.build_request("a cat in a spacesuit", ImageOptions())
    assert req.steps == PROFILES["l4"].image_default_steps
    assert (req.width, req.height) == size_for("1:1", 1024)
    assert not req.is_edit


def test_empty_prompt_rejected():
    engine, *_ = _engine()
    with pytest.raises(ValueError):
        engine.build_request("   ", ImageOptions())


def test_generation_swaps_chat_out_on_low_vram():
    engine, mm, chat, image = _engine("l4")
    mm.ensure("chat")
    img = engine.run(engine.build_request("x", ImageOptions(steps=3, seed=1)))
    assert img.size == size_for("1:1", 1024)
    assert mm.loaded_models() == ["image"]


def test_progress_and_cancel():
    engine, *_ = _engine()
    seen = []
    engine.run(engine.build_request("x", ImageOptions(steps=4)), progress=lambda s, t: seen.append((s, t)))
    assert seen[-1] == (4, 4)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        engine.run(engine.build_request("x", ImageOptions(steps=4)), cancel=cancel)


def test_edit_never_reuses_parent_seed():
    from PIL import Image

    engine, *_ = _engine()
    src = Image.new("RGB", (640, 360))
    req = engine.build_edit_request("brighter", [src], ImageOptions(seed=42), parent_seeds={42})
    assert req.seed != 42
    assert req.is_edit
    assert req.width is None and req.height is None  # aspect follows the source image


def test_edit_requires_source_and_caps_condition_images():
    from PIL import Image

    engine, *_ = _engine()
    with pytest.raises(ValueError):
        engine.build_edit_request("x", [], ImageOptions())
    req = engine.build_edit_request("x", [Image.new("RGB", (8, 8))] * 12, ImageOptions())
    assert len(req.images) == 10


def test_edit_runs_on_mock_backend():
    from PIL import Image

    engine, _, _, image = _engine()
    src = Image.new("RGB", (64, 64), (255, 255, 255))
    out = engine.run(engine.build_edit_request("darker", [src], ImageOptions(steps=2)))
    assert out.size == (512, 512)
    assert image.requests[-1].is_edit


def test_configured_defaults_presets_and_limits():
    from qmc.image_engine import preset_values

    engine, mm, *_ = _engine("a100_80")
    engine = ImageEngine(mm, engine.profile, default_band=1536, default_steps=32, max_band=1536)
    request = engine.build_request("cat", ImageOptions())
    assert (request.output_resolution, request.steps) == (1536, 32)
    assert preset_values("configured", engine) == (1536, 32, 1)
    assert preset_values("draft", engine) == (768, 20, 1)
    assert preset_values("quality", engine) == (1536, 50, 1)
    assert engine.build_request("cat", ImageOptions(band=2048)).output_resolution == 1536
    full, *_ = _engine("a100_80")
    assert preset_values("quality", full) == (2048, 50, 1)


def test_cancelled_request_does_not_load_model():
    engine, mm, _, _ = _engine()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        engine.run(engine.build_request("cat", ImageOptions()), cancel=cancel)
    assert not mm.events
