"""Qwen-Image-2.1 (diffusers ``QwenImage21Pipeline``) as a ManagedModel + ImageBackend.

Precision / placement follow the working ``Qwen-Image-2.1-Colab-Pro.ipynb``:
- bf16 on A100; on L4 the DiT is torchao int8 weight-only, text encoder stays bf16
- ``enable_model_cpu_offload()`` below 70 GiB (text_encoder -> transformer -> vae swap in/out)
- the Qwen-Image-2.1 VAE is loaded explicitly in bf16
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from PIL import Image, ImageOps

from ..config import AppConfig
from ..gpu_manager import GPUProfile, free_cuda_memory
from ..imaging import flatten_alpha
from .base import Cancelled, ImageRequest, ProgressFn

log = logging.getLogger(__name__)


def _from_pretrained(cls, *args, dtype=None, **kwargs):
    """diffusers is migrating ``torch_dtype`` -> ``dtype``; support both."""
    try:
        return cls.from_pretrained(*args, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(*args, torch_dtype=dtype, **kwargs)


class QwenImageModel:
    name = "image"

    def __init__(self, cfg: AppConfig, profile: GPUProfile):
        self.cfg = cfg
        self.profile = profile
        self.precision = profile.image_precision if cfg.image.precision == "auto" else cfg.image.precision
        self.placement = profile.image_placement
        self.force_vae_tiling = False
        self._pipe: Any = None
        self._active = False
        self._offload_enabled = False

    @property
    def label(self) -> str:
        return f"Qwen-Image-2.1 (DiT {self.precision}, {self.placement})"

    @property
    def model_label(self) -> str:
        return self.label

    # ------------------------------------------------------------------ ManagedModel
    @property
    def is_loaded(self) -> bool:
        return self._active and self._pipe is not None

    @property
    def uses_local_gpu(self) -> bool:
        return True

    def _build(self) -> Any:
        import torch  # noqa: PLC0415
        from diffusers import QwenImage21Pipeline  # noqa: PLC0415

        try:
            from diffusers import AutoencoderKLQwenImage21  # noqa: PLC0415
        except ImportError:  # older layout
            from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (  # noqa: PLC0415
                AutoencoderKLQwenImage21,
            )

        model_id = self.cfg.image.model_id
        cache = str(self.cfg.hf_cache_dir) if self.cfg.hf_cache_dir else None
        vae = _from_pretrained(
            AutoencoderKLQwenImage21, model_id, subfolder="vae", dtype=torch.bfloat16, cache_dir=cache
        )
        quant_config = None
        if self.precision == "int8":
            try:
                from diffusers import PipelineQuantizationConfig  # noqa: PLC0415
                from diffusers import TorchAoConfig as DiffusersTorchAoConfig  # noqa: PLC0415
                from torchao.quantization import Int8WeightOnlyConfig  # noqa: PLC0415

                quant_config = PipelineQuantizationConfig(
                    quant_mapping={"transformer": DiffusersTorchAoConfig(quant_type=Int8WeightOnlyConfig())}
                )
            except Exception as exc:
                log.warning("int8 setup failed (%s); falling back to bf16 + model offload", exc)
                self.precision = "bf16"
                self.placement = "model_offload"
        return _from_pretrained(
            QwenImage21Pipeline,
            model_id,
            vae=vae,
            dtype=torch.bfloat16,
            quantization_config=quant_config,
            cache_dir=cache,
        )

    def load(self) -> None:
        if self._pipe is None:
            try:
                self._pipe = self._build()
            except Exception as exc:
                self._pipe = None
                raise RuntimeError(
                    f"Qwen-Image-2.1 のロードに失敗しました: {exc}\n"
                    "（初回は約40GBのダウンロードが必要です。HF_TOKEN / ディスク空き / diffusers のバージョンを確認）"
                ) from exc
            self._offload_enabled = False
        if self.placement == "gpu":
            self._pipe.to("cuda")
        elif not self._offload_enabled:
            self._pipe.enable_model_cpu_offload()
            self._offload_enabled = True
        self._active = True

    def unload(self) -> None:
        if self._pipe is None:
            self._active = False
            return
        if self.profile.keep_image_in_ram:
            if self.placement == "gpu":
                self._pipe.to("cpu")
            elif hasattr(self._pipe, "maybe_free_model_hooks"):
                self._pipe.maybe_free_model_hooks()  # offload every component back to CPU
        else:
            self._pipe = None
            self._offload_enabled = False
        self._active = False
        free_cuda_memory()

    def degrade(self) -> bool:
        """OOM fallback ladder: GPU-resident -> model offload -> VAE tiling -> int8 DiT."""
        if self.placement == "gpu":
            self.placement = "model_offload"
            if self._pipe is not None:
                self._pipe.to("cpu")
            self._active = False
            return True
        if not self.force_vae_tiling:
            self.force_vae_tiling = True
            return True
        if self.precision == "bf16":
            self.precision = "int8"
            self._pipe = None  # rebuild with the quantized DiT
            self._offload_enabled = False
            self._active = False
            return True
        return False

    # ------------------------------------------------------------------ ImageBackend
    def generate(
        self, request: ImageRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> Image.Image:
        import torch  # noqa: PLC0415

        pipe = self._pipe
        if pipe is None:
            raise RuntimeError("image model is not loaded")
        band = max(request.width or 0, request.height or 0, request.output_resolution)
        tiling = self.force_vae_tiling or (
            self.profile.vae_tiling_band and band >= self.profile.vae_tiling_band
        )
        if hasattr(pipe.vae, "enable_tiling"):
            if tiling:
                pipe.vae.enable_tiling()
            elif hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()

        def on_step_end(p, step, timestep, cb_kwargs):
            if progress:
                progress(step + 1, request.steps)
            if cancel is not None and cancel.is_set():
                p._interrupt = True
            return cb_kwargs

        prompt = request.prompt
        condition_images = list(request.images)
        if request.mask_image is not None and condition_images:
            # The pinned QwenImage21Pipeline has no mask_image argument. Provide a visual hint
            # and instruction instead; this does not guarantee pixel-perfect preservation.
            source = condition_images[-1].convert("RGB")
            mask = request.mask_image.convert("L").resize(source.size)
            red = Image.new("RGB", source.size, (240, 40, 85))
            overlay = Image.blend(source, Image.composite(red, source, mask), 0.55)
            condition_images = condition_images[:-1][:7] + [
                overlay,
                ImageOps.colorize(mask, "black", "white"),
                source,
            ]
            prompt = "Edit ONLY the masked region. Keep unmasked pixels unchanged. " + prompt
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": request.steps,
            "output_resolution": request.output_resolution,
            "generator": torch.Generator("cuda").manual_seed(request.seed),
            "callback_on_step_end": on_step_end,
            "true_cfg_scale": 1.0,
        }
        if request.width and request.height:
            kwargs.update(width=request.width, height=request.height)
        if condition_images:
            kwargs["image"] = condition_images
        if request.negative_prompt and request.true_cfg_scale > 1.0:
            kwargs.update(negative_prompt=request.negative_prompt, true_cfg_scale=request.true_cfg_scale)

        free_cuda_memory()
        with torch.inference_mode():
            result = pipe(**kwargs).images[0]
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        return flatten_alpha(result)
