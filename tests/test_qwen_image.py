"""Exercise the real backend wrapper without CUDA, diffusers, or model weights."""

import contextlib
import sys
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from qmc.backends import qwen_image
from qmc.backends.base import Cancelled, ImageRequest
from qmc.config import AppConfig
from qmc.gpu_manager import PROFILES


class FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakePipeline:
    def __init__(self):
        self.tiling = []
        self.vae = SimpleNamespace(
            enable_tiling=lambda: self.tiling.append(True),
            disable_tiling=lambda: self.tiling.append(False),
        )
        self.calls = []
        self.decode_calls = 0
        self.cleanup_calls = 0
        self.call_error = None
        self.cleanup_error = None
        self.after_decode = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.call_error is not None:
            raise self.call_error
        for step in range(kwargs["num_inference_steps"]):
            callback_kwargs = {"latents": object()}
            result = kwargs["callback_on_step_end"](self, step, step, callback_kwargs)
            assert result is callback_kwargs
        self.decode_calls += 1
        if self.after_decode:
            self.after_decode()
        # The pinned pipeline performs its own cleanup on the successful path.
        self.maybe_free_model_hooks()
        return SimpleNamespace(images=[Image.new("RGBA", (8, 8), (10, 20, 30, 255))])

    def maybe_free_model_hooks(self):
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


@pytest.fixture
def backend(monkeypatch):
    generators = []

    def generator(device):
        result = FakeGenerator(device)
        generators.append(result)
        return result

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(Generator=generator, inference_mode=contextlib.nullcontext),
    )

    def unexpected_flush():
        pytest.fail("generation must not flush the CUDA allocator")

    monkeypatch.setattr(qwen_image, "free_cuda_memory", unexpected_flush)
    model = qwen_image.QwenImageModel(AppConfig(), PROFILES["a100_80"])
    pipe = FakePipeline()
    model._pipe = pipe
    model._active = True
    return model, pipe, generators


def test_pre_cancel_skips_pipeline_and_cuda_generator(backend):
    model, pipe, generators = backend
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(Cancelled):
        model.generate(ImageRequest(prompt="cat"), cancel=cancel)

    assert not pipe.calls
    assert not pipe.tiling
    assert not generators
    assert pipe.cleanup_calls == 0


def test_step_cancel_skips_decode_cleans_hooks_and_allows_next_generation(backend):
    model, pipe, _ = backend
    cancel = threading.Event()
    progress = []

    def stop(step, total):
        progress.append((step, total))
        cancel.set()

    request = ImageRequest(prompt="cat", steps=3)
    with pytest.raises(Cancelled):
        model.generate(request, progress=stop, cancel=cancel)

    assert progress == [(1, 3)]
    assert pipe.decode_calls == 0
    assert pipe.cleanup_calls == 1
    cancel.clear()
    result = model.generate(request, cancel=cancel)
    assert result.mode == "RGB"
    assert pipe.decode_calls == 1
    assert pipe.cleanup_calls == 2  # one failure cleanup, one normal pipeline cleanup


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), ValueError("bad input"), KeyboardInterrupt()])
def test_pipeline_exception_cleans_hooks_and_preserves_exception(backend, error):
    model, pipe, _ = backend
    pipe.call_error = error

    with pytest.raises(type(error)) as caught:
        model.generate(ImageRequest(prompt="cat"))

    assert caught.value is error
    assert pipe.cleanup_calls == 1
    assert pipe.decode_calls == 0


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), Cancelled()])
def test_cleanup_failure_does_not_replace_original_exception(backend, caplog, error):
    model, pipe, _ = backend
    pipe.call_error = error
    pipe.cleanup_error = RuntimeError("cleanup failed")

    with pytest.raises(type(error)) as caught:
        model.generate(ImageRequest(prompt="cat"))

    assert caught.value is error
    assert pipe.cleanup_calls == 1
    assert "cleanup failed" in caplog.text


@pytest.mark.parametrize(
    ("negative", "cfg", "expected_cfg"),
    [(None, 2.0, 1.0), ("", 2.0, 1.0), ("blurry", 1.0, 1.0), ("blurry", 2.0, 2.0)],
)
def test_generation_parameters_guidance_and_no_allocator_flush(backend, negative, cfg, expected_cfg):
    model, pipe, generators = backend
    progress = []
    request = ImageRequest(
        prompt="a cat",
        width=768,
        height=512,
        output_resolution=512,
        steps=3,
        seed=123,
        negative_prompt=negative,
        true_cfg_scale=cfg,
    )

    result = model.generate(request, progress=lambda step, total: progress.append((step, total)))

    kwargs = pipe.calls[0]
    assert kwargs["prompt"] == "a cat"
    assert kwargs["num_inference_steps"] == 3
    assert (kwargs["width"], kwargs["height"], kwargs["output_resolution"]) == (768, 512, 512)
    assert kwargs["true_cfg_scale"] == expected_cfg
    if expected_cfg > 1:
        assert kwargs["negative_prompt"] == negative
    else:
        assert "negative_prompt" not in kwargs
    assert "image" not in kwargs
    assert (generators[0].device, generators[0].seed) == ("cuda", 123)
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert (result.size, result.mode) == ((8, 8), "RGB")
    assert pipe.cleanup_calls == 1  # wrapper must not repeat successful pipeline cleanup


def test_edit_preserves_condition_images_and_automatic_dimensions(backend):
    model, pipe, _ = backend
    sources = [Image.new("RGB", (24, 16))]

    model.generate(ImageRequest(prompt="make it brighter", images=sources, steps=1))

    assert pipe.calls[0]["image"] == sources
    assert "width" not in pipe.calls[0]
    assert "height" not in pipe.calls[0]


def test_cancel_during_decode_discards_result_without_repeating_cleanup(backend):
    model, pipe, _ = backend
    cancel = threading.Event()
    pipe.after_decode = cancel.set

    with pytest.raises(Cancelled):
        model.generate(ImageRequest(prompt="cat", steps=1), cancel=cancel)

    assert pipe.decode_calls == 1
    assert pipe.cleanup_calls == 1
