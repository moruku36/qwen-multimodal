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


def render_pdf_pages(
    path: str | Path, max_pages: int = 6, max_side: int = 2048, max_mb: int = 30
) -> list[Image.Image]:
    """Render the first PDF pages without a system poppler install."""
    p = Path(path)
    if p.stat().st_size > max_mb * 1024**2:
        raise InvalidImageError(f"PDFが大きすぎます（上限 {max_mb}MB）: {p.name}")
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise InvalidImageError("PDFサポートの依存が入っていません（pypdfium2）") from exc
    try:
        pdf = pdfium.PdfDocument(str(p))
        pages = []
        try:
            for i in range(min(len(pdf), max_pages)):
                page = pdf[i]
                scale = min(2.0, max_side / max(page.get_size()))
                bitmap = page.render(scale=scale)
                pages.append(fit_max_side(bitmap.to_pil().convert("RGB"), max_side))
                page.close()
        finally:
            pdf.close()
        if not pages:
            raise InvalidImageError("PDFにページがありません")
        return pages
    except Exception as exc:
        raise InvalidImageError(f"PDFを読み込めません: {p.name} ({exc})") from exc


def video_sample_times(duration: float, max_frames: int = 8) -> list[float]:
    """Even samples including the first and last decodable moments."""
    if duration <= 0:
        raise InvalidImageError("動画の長さを取得できません")
    count = min(max_frames, max(1, int(duration) + 1))
    if count == 1:
        return [0.0]
    return [i * max(0, duration - 0.1) / (count - 1) for i in range(count)]


def webm_has_video(path: str | Path) -> bool:
    import json
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return False  # microphone uploads still work without the video dependency
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return bool(json.loads(probe.stdout).get("streams"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def sample_video_frames(
    path: str | Path,
    max_frames: int = 8,
    max_side: int = 768,
    max_mb: int = 80,
    max_duration_s: float = 30,
) -> tuple[list[Image.Image], float]:
    """Probe before decoding; return RGB frames sampled across a short video."""
    import json
    import shutil
    import subprocess

    p = Path(path)
    if p.stat().st_size > max_mb * 1024**2:
        raise InvalidImageError(f"動画が大きすぎます（上限 {max_mb}MB）")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise InvalidImageError(
            "ffmpeg がありません。Colab の ffmpeg を確認するか ffmpeg をインストールしてください"
        )
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(p)],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        if duration > max_duration_s:
            raise InvalidImageError(f"動画は{max_duration_s:g}秒以内にしてください")
        frames = []
        for second in video_sample_times(duration, max_frames):
            for offset in (0, 0.5, 1.0):
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-ss",
                        str(max(0, second - offset)),
                        "-i",
                        str(p),
                        "-frames:v",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "pipe:1",
                    ],
                    capture_output=True,
                    timeout=30,
                    check=True,
                )
                if result.stdout:
                    with Image.open(io.BytesIO(result.stdout)) as image:
                        frames.append(fit_max_side(image.convert("RGB"), max_side))
                    break
            else:
                raise InvalidImageError("動画からフレームを取り出せません")
        return frames, duration
    except InvalidImageError:
        raise
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as exc:
        raise InvalidImageError(f"動画を読み込めません: {p.name} ({exc})") from exc


def mask_from_editor(editor: dict | None, size: tuple[int, int]) -> Image.Image | None:
    """Use painted layer alpha as a single-channel mask at the source image size."""
    from PIL import ImageChops, ImageFilter

    if not isinstance(editor, dict):
        return None
    mask = Image.new("L", size, 0)
    for layer in editor.get("layers") or []:
        if layer is None:
            continue
        if not isinstance(layer, Image.Image):
            with Image.open(layer) as opened:
                layer = opened.copy()
        alpha = layer.getchannel("A") if "A" in layer.getbands() else layer.convert("L")
        mask = ImageChops.lighter(mask, alpha.resize(size, Image.Resampling.NEAREST))
    if not mask.getbbox():
        return None
    return mask.point(lambda pixel: 255 if pixel > 12 else 0).filter(ImageFilter.MaxFilter(9))


def slice_tall_image(
    path: str | Path, max_side: int = 2048, overlap: int = 128, max_bands: int = 6
) -> list[Image.Image]:
    """Return overlapping readable bands of a tall screenshot, or an empty list."""
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        width, height = source.size
        if height <= max_side or height / width < 2.2:
            return []
        if width > max_side:
            source = source.resize((max_side, round(height * max_side / width)), Image.LANCZOS)
        source = source.convert("RGB")
        step = max(1, max_side - overlap)
        starts = list(range(0, max(1, source.height - max_side + 1), step))
        last = max(0, source.height - max_side)
        if not starts or starts[-1] != last:
            starts.append(last)
        return [
            source.crop((0, y, source.width, min(y + max_side, source.height))) for y in starts[:max_bands]
        ]


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
