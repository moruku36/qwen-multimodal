"""Image generation and editing orchestration (backend-neutral)."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass

from PIL import Image

from .backends.base import ImageRequest, ProgressFn
from .gpu_manager import GPUProfile
from .model_manager import IMAGE, ModelManager

# Official Qwen-Image-2.1 recommended sizes at the 2048 band (HF model card).
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
BANDS = (512, 768, 1024, 1280, 1536, 2048)
# Always-on negative prompt. The backend only applies it when true_cfg_scale > 1.
DEFAULT_NEGATIVE_PROMPT = (
    "child, children, kid, toddler, infant, baby, minor, underage, teenager, schoolchild, "
    "childlike body, loli, shota, low quality, blurry, deformed, extra fingers, watermark"
)
DEFAULT_TRUE_CFG_SCALE = 2.0  # docs/phase0-research.md: ~2 when a negative prompt is used

MAX_CONDITION_IMAGES = 10  # Qwen-Image-2.1 model card


def size_for(aspect: str, band: int) -> tuple[int, int]:
    """Scale the official 2048-band size to ``band`` and round to multiples of 32."""
    w, h = ASPECT_RATIOS.get(aspect, ASPECT_RATIOS["1:1"])
    scale = band / 2048
    return round(w * scale / 32) * 32, round(h * scale / 32) * 32


def clamp_band(band: int, profile: GPUProfile) -> int:
    return max(512, min(int(band), profile.image_max_band))


def choose_seed(requested: int | None, avoid: set[int] | None = None) -> int:
    """Random seed unless given. Never reuse a parent's seed for an edit: same seed + same size
    makes Qwen-Image-2.1 return a near-copy that ignores the instruction (diffusers#14824)."""
    avoid = avoid or set()
    seed = requested if requested is not None and requested >= 0 else random.randrange(2**31)
    while seed in avoid:
        seed = (seed + 7919) % (2**31)
    return seed


@dataclass
class ImageOptions:
    aspect: str = "1:1"
    band: int | None = None
    steps: int | None = None
    seed: int | None = None  # None / -1 = random
    negative_prompt: str | None = None
    true_cfg_scale: float = DEFAULT_TRUE_CFG_SCALE
    variations: int = 1


class ImageEngine:
    def __init__(self, manager: ModelManager, profile: GPUProfile, default_band: int = 1024):
        self.manager = manager
        self.profile = profile
        self.default_band = default_band

    def build_request(
        self,
        prompt: str,
        options: ImageOptions,
        sources: list[Image.Image] | None = None,
        avoid_seeds: set[int] | None = None,
    ) -> ImageRequest:
        if not prompt or not prompt.strip():
            raise ValueError("プロンプトが空です。")
        band = clamp_band(options.band or self.default_band, self.profile)
        steps = int(options.steps or self.profile.image_default_steps)
        seed = choose_seed(options.seed, avoid_seeds)
        negative = options.negative_prompt
        if negative is None:
            negative = DEFAULT_NEGATIVE_PROMPT
        request = ImageRequest(
            prompt=prompt.strip(),
            images=list(sources or []),
            output_resolution=band,
            steps=max(1, min(steps, 100)),
            seed=seed,
            negative_prompt=negative,
            true_cfg_scale=options.true_cfg_scale,
        )
        if not request.images:
            request.width, request.height = size_for(options.aspect, band)
        # For edits width/height stay None: the pipeline keeps the source aspect ratio.
        return request

    def build_edit_request(
        self,
        instruction: str,
        sources: list[Image.Image],
        options: ImageOptions,
        parent_seeds: set[int] | None = None,
    ) -> ImageRequest:
        """Image-to-image edit. ``sources`` order matters (the last one is the image being edited
        and decides the output aspect ratio); at most 10 condition images are supported."""
        if not sources:
            raise ValueError("編集する画像がありません。画像を添付するか、先に画像を生成してください。")
        if len(sources) > MAX_CONDITION_IMAGES:
            sources = sources[-MAX_CONDITION_IMAGES:]
        return self.build_request(instruction, options, sources=sources, avoid_seeds=parent_seeds)

    def run(
        self, request: ImageRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> Image.Image:
        return self.manager.run(IMAGE, lambda model: model.generate(request, progress, cancel))
