"""Small documented HTTP adapter for a remote image worker."""

from __future__ import annotations

import base64
import io
import threading
from urllib.parse import urlparse

import requests
from PIL import Image

from .base import Cancelled, ImageRequest, ProgressFn


class HttpImageModel:
    name = "image"

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._loaded = False
        host = urlparse(base_url).netloc or base_url
        self.label = f"Image: remote ({host})"
        self.model_label = self.label

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def uses_local_gpu(self) -> bool:
        return False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def degrade(self) -> bool:
        return False

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def _png_bytes(image: Image.Image) -> bytes:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def generate(
        self, request: ImageRequest, progress: ProgressFn | None = None, cancel: threading.Event | None = None
    ) -> Image.Image:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        endpoint = "edits" if request.is_edit else "generations"
        url = f"{self.base_url}/v1/images/{endpoint}"
        fields = {
            "prompt": request.prompt,
            "width": request.width or request.output_resolution,
            "height": request.height or request.output_resolution,
            "steps": request.steps,
            "seed": request.seed,
            "negative_prompt": request.negative_prompt or "",
        }
        try:
            if request.is_edit:
                files = [
                    ("images", (f"reference-{index}.png", self._png_bytes(image), "image/png"))
                    for index, image in enumerate(request.images, 1)
                ]
                if request.mask_image is not None:
                    files.append(("mask", ("mask.png", self._png_bytes(request.mask_image), "image/png")))
                response = requests.post(
                    url, headers=self._headers(), data=fields, files=files, timeout=(10, self.timeout_s)
                )
            else:
                response = requests.post(
                    url, headers=self._headers(), json=fields, timeout=(10, self.timeout_s)
                )
            response.raise_for_status()
            payload = response.json()
            item = payload["data"][0]
            if "b64_json" in item:
                raw = base64.b64decode(item["b64_json"])
            elif "url" in item:
                image_url = item["url"]
                if image_url.startswith("data:"):
                    raw = base64.b64decode(image_url.split(",", 1)[1])
                else:
                    parsed = urlparse(image_url)
                    if parsed.scheme not in {"https", "http"}:
                        raise ValueError("画像URLの形式が不正です")
                    download = requests.get(image_url, timeout=(10, self.timeout_s))
                    download.raise_for_status()
                    raw = download.content
            else:
                raise ValueError("画像データがありません")
            if len(raw) > 80 * 1024**2:
                raise ValueError("返された画像が大きすぎます")
            with Image.open(io.BytesIO(raw)) as image:
                result = image.convert("RGB")
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            return result
        except requests.RequestException as exc:
            raise RuntimeError(f"リモート画像APIに接続できません（{endpoint}）: {exc}") from exc
        except (KeyError, IndexError, ValueError, OSError) as exc:
            raise RuntimeError(f"リモート画像APIの応答を読み込めません: {exc}") from exc
