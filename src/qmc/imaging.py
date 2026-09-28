"""Image validation / normalization helpers (CPU only)."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 100_000_000  # refuse decompression bombs beyond ~100MP

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF", "MPO"}


class InvalidImageError(ValueError):
    pass


def load_user_image(path: str | Path, max_side: int = 2048, max_mb: int = 30) -> tuple[Image.Image, dict]:
    """Open + validate an uploaded image. Returns (RGB/RGBA image, info).

    - rejects non-images, unsupported formats, files over ``max_mb``
    - applies EXIF orientation
    - downsizes so the long side is <= ``max_side`` (huge images)
    """
    p = Path(path)
    if not p.exists():
        raise InvalidImageError(f"ファイルが見つかりません: {p.name}")
    size_mb = p.stat().st_size / 1024**2
    if size_mb > max_mb:
        raise InvalidImageError(f"画像が大きすぎます（{size_mb:.1f}MB > {max_mb}MB）: {p.name}")
    try:
        with Image.open(p) as probe:
            fmt = probe.format
            probe.verify()
        img = Image.open(p)
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"画像として読み込めません: {p.name} ({exc})") from exc
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(f"未対応の画像形式です: {fmt}")
    img = ImageOps.exif_transpose(img)
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") and _has_alpha(img) else img.convert("RGB")
    original = img.size
    img = fit_max_side(img, max_side)
    return img, {
        "format": fmt,
        "original_size": original,
        "size": img.size,
        "downscaled": original != img.size,
    }


def _has_alpha(img: Image.Image) -> bool:
    if img.mode == "P":
        return "transparency" in img.info
    return "A" in img.getbands()


def fit_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def flatten_alpha(img: Image.Image) -> Image.Image:
    """RGBA -> RGB. Keeps RGBA only if the alpha channel is actually used."""
    if img.mode != "RGBA":
        return img.convert("RGB") if img.mode != "RGB" else img
    alpha = img.getchannel("A")
    if alpha.getextrema() == (255, 255):
        return img.convert("RGB")
    return img


def to_data_uri(path: str | Path, max_side: int = 1280) -> str:
    """Encode an image file as a data URI for the OpenAI-compatible vision API."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = fit_max_side(img, max_side)
        buf = io.BytesIO()
        if img.mode == "RGBA":
            img.save(buf, format="PNG")
            mime = "image/png"
        else:
            img.convert("RGB").save(buf, format="JPEG", quality=92)
            mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def save_png(img: Image.Image, path: Path) -> str:
    """Save atomically (tmp + rename; safe on Drive) and return sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    img.save(tmp, format="PNG")
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(path)
    return digest
