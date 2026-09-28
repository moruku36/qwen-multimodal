"""Qwen3.8-27B (text + vision) via a local llama.cpp ``llama-server`` subprocess.

Unloading = stopping the process, which releases 100% of its VRAM (no allocator leftovers).
A remote OpenAI-compatible server (vLLM on RunPod, etc.) is supported by ``RemoteChatModel``.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from ..config import AppConfig
from ..gpu_manager import GPUProfile
from .base import ChatDelta, ChatParams
from .openai_compat import OpenAICompatClient

log = logging.getLogger(__name__)

OOM_MARKERS = ("out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate")


def find_llama_server(bin_dir: Path | None = None) -> Path | None:
    candidates = []
    if bin_dir:
        candidates.append(Path(bin_dir) / "llama-server")
    candidates += [
        Path("/content/llama.cpp/build/bin/llama-server"),
        Path.home() / "llama.cpp/build/bin/llama-server",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    found = shutil.which("llama-server")
    return Path(found) if found else None


def download_hf_file(repo_id: str, filename: str, cache_dir: Path | None = None, retries: int = 3) -> Path:
    """hf_hub_download with retries and a readable error (network / auth / disk)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id, filename=filename, cache_dir=str(cache_dir) if cache_dir else None
                )
            )
        except Exception as exc:  # network errors, 401/403, disk full
            last = exc
            log.warning(
                "HF download failed (%s/%s) attempt %d/%d: %s", repo_id, filename, attempt, retries, exc
            )
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(
        f"Hugging Face からのダウンロードに失敗しました: {repo_id}/{filename}\n"
        f"原因: {last}\n対処: ネットワーク / HF_TOKEN（Colab Secrets）/ ディスク空き容量を確認してください。"
    )


def build_command(
    server: Path,
    model_path: Path,
    mmproj_path: Path | None,
    *,
    host: str,
    port: int,
    ctx_size: int,
    fit_target_mib: int,
    api_key: str | None,
    image_max_tokens: int | None = None,
) -> list[str]:
    cmd = [
        str(server),
        "-m", str(model_path),
        "--fit", "on",
        "--fit-target", str(fit_target_mib),
        "-c", str(ctx_size),
        "-np", "1",  # single user: one slot keeps the KV cache small
        "--jinja",
        "--reasoning-format", "deepseek",  # thoughts -> reasoning_content
        "--host", host,
        "--port", str(port),
        "--no-webui",
    ]  # fmt: skip
    if mmproj_path:
        cmd += ["--mmproj", str(mmproj_path)]
    if image_max_tokens:
        cmd += ["--image-max-tokens", str(image_max_tokens)]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


class LlamaServerModel:
    """ManagedModel + ChatBackend for a local llama-server process."""

    name = "chat"

    def __init__(self, cfg: AppConfig, profile: GPUProfile):
        self.cfg = cfg
        self.label = f"{cfg.chat.hf_repo}/{cfg.chat.model_file} + mmproj ({cfg.chat.mmproj_repo or cfg.chat.hf_repo})"
        self.profile = profile
        self.ctx_size = cfg.chat.ctx_size or profile.chat_ctx_size
        self.fit_target = cfg.chat.fit_target_mib or profile.chat_fit_target_mib
        self.proc: subprocess.Popen | None = None
        self.api_key = secrets.token_urlsafe(24)  # random per launch; server is bound to localhost anyway
        self.log_path = Path(cfg.local_db_path).parent / "llama-server.log"
        self._paths: tuple[Path, Path | None] | None = None
        self.client = OpenAICompatClient(
            f"http://{cfg.chat.host}:{cfg.chat.port}/v1",
            api_key=self.api_key,
            model="qwen3.8-27b",
            timeout=cfg.chat.request_timeout_s,
        )

    # ---- ManagedModel
    @property
    def is_loaded(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def uses_local_gpu(self) -> bool:
        return True

    def resolve_files(self) -> tuple[Path, Path | None]:
        if self._paths is None:
            c = self.cfg.chat
            model = download_hf_file(c.hf_repo, c.model_file, self.cfg.hf_cache_dir)
            mmproj_repo = c.mmproj_repo or c.hf_repo
            mmproj = download_hf_file(mmproj_repo, c.mmproj_file, self.cfg.hf_cache_dir) if c.mmproj_file else None
            self._paths = (model, mmproj)
        return self._paths

    def load(self) -> None:
        server = find_llama_server(self.cfg.llama_bin_dir)
        if server is None:
            raise RuntimeError(
                "llama-server が見つかりません。Notebook の依存関係セル（llama.cpp ビルド）を実行してください。"
            )
        model_path, mmproj_path = self.resolve_files()
        cmd = build_command(
            server,
            model_path,
            mmproj_path,
            host=self.cfg.chat.host,
            port=self.cfg.chat.port,
            ctx_size=self.ctx_size,
            fit_target_mib=self.fit_target,
            api_key=self.api_key,
            image_max_tokens=self.cfg.chat.image_max_tokens,
        )
        self._kill_stale()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "w")  # noqa: SIM115 - owned by the child process lifetime
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{server.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        self.proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env
        )
        deadline = time.time() + self.cfg.chat.startup_timeout_s
        health = f"http://{self.cfg.chat.host}:{self.cfg.chat.port}/health"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self.log_tail()
                self.proc = None
                if any(m in tail.lower() for m in OOM_MARKERS):
                    raise RuntimeError(f"CUDA out of memory while starting llama-server:\n{tail}")
                raise RuntimeError(f"llama-server の起動に失敗しました:\n{tail}")
            try:
                if requests.get(health, timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        self.unload()
        raise RuntimeError(f"llama-server が {self.cfg.chat.startup_timeout_s}s 以内に起動しませんでした")

    def unload(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None

    def degrade(self) -> bool:
        """Halve the context window (min 4096) and leave more VRAM headroom."""
        changed = False
        if self.ctx_size > 4096:
            self.ctx_size = max(4096, self.ctx_size // 2)
            changed = True
        if self.fit_target < 4096:
            self.fit_target += 1024
            changed = True
        return changed

    def _kill_stale(self) -> None:
        """Stop a llama-server left over from a previous run of the notebook (same port)."""
        subprocess.run(
            ["pkill", "-f", f"llama-server.*--port {self.cfg.chat.port}"],
            capture_output=True,
            check=False,
        )
        time.sleep(1)

    def log_tail(self, n: int = 3000) -> str:
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except OSError:
            return ""

    # ---- ChatBackend
    def stream_chat(
        self, messages: list[dict], params: ChatParams, cancel: threading.Event | None = None
    ) -> Iterator[ChatDelta]:
        return self.client.stream_chat(messages, params, cancel)


class RemoteChatModel:
    """OpenAI-compatible remote server (vLLM / RunPod / cloud). Uses no local VRAM."""

    name = "chat"

    def __init__(self, cfg: AppConfig):
        c = cfg.chat
        self.label = f"{c.remote_model_name} @ {c.remote_base_url}"
        self.client = OpenAICompatClient(
            c.remote_base_url or "",
            api_key=c.remote_api_key,
            model=c.remote_model_name,
            timeout=c.request_timeout_s,
        )
        self._ok = False

    @property
    def is_loaded(self) -> bool:
        return self._ok

    @property
    def uses_local_gpu(self) -> bool:
        return False

    def load(self) -> None:
        try:
            r = requests.get(f"{self.client.base_url}/models", headers=self.client._headers(), timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"リモートChat APIに接続できません: {exc}") from exc
        self._ok = True

    def unload(self) -> None:
        self._ok = False

    def degrade(self) -> bool:
        return False

    def stream_chat(
        self, messages: list[dict], params: ChatParams, cancel: threading.Event | None = None
    ) -> Iterator[ChatDelta]:
        return self.client.stream_chat(messages, params, cancel)
