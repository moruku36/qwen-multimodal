"""Google Colab helpers used by the thin notebook (``Qwen-Multimodal-Colab.ipynb``).

Every function works (as a no-op or with a clear message) outside Colab, so the logic is
testable and the notebook only calls: ``setup()`` -> ``install_llama_cpp()`` -> ``launch()``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# llama.cpp commit verified to have every flag we use (see docs/phase0-research.md)
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_COMMIT = "4da6337767f973e2b4d0797e5b323d77d8565e4a"
# 80 = A100, 89 = L4, 90 = H100. Building all lets one Drive cache serve every Colab GPU.
CUDA_ARCHS = "80;89;90"
DRIVE_ROOT = Path("/content/drive/MyDrive")
APP_DRIVE_DIR = DRIVE_ROOT / "qwen-multimodal-colab"
SECRET_NAMES = ("HF_TOKEN", "QMC_AUTH_USER", "QMC_AUTH_PASSWORD", "QMC_CHAT_API_KEY", "QMC_CHAT_BASE_URL")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def mount_drive(mountpoint: str = "/content/drive") -> bool:
    """Mount Google Drive. Returns False (history is then kept locally only) if not possible."""
    if DRIVE_ROOT.exists():
        return True
    if not in_colab():
        return False
    try:
        from google.colab import drive  # noqa: PLC0415

        drive.mount(mountpoint)
        return DRIVE_ROOT.exists()
    except Exception as exc:
        print(f"⚠️ Google Drive をマウントできませんでした（履歴はランタイム終了で消えます）: {exc}")
        return False


def load_secrets(names: tuple[str, ...] = SECRET_NAMES) -> list[str]:
    """Copy Colab Secrets into environment variables. Never prints values."""
    loaded = []
    if not in_colab():
        return loaded
    from google.colab import userdata  # noqa: PLC0415

    for name in names:
        if os.environ.get(name):
            loaded.append(name)
            continue
        try:
            value = userdata.get(name)
        except Exception:  # SecretNotFoundError / NotebookAccessError
            continue
        if value:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def setup(use_drive: bool = True) -> dict:
    """Cell 3: Drive mount + secrets + data directory."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    drive_ok = mount_drive() if use_drive else False
    if drive_ok:
        os.environ.setdefault("QMC_DATA_DIR", str(APP_DRIVE_DIR / "data"))
    secrets = load_secrets()
    info = {
        "drive": drive_ok,
        "data_dir": os.environ.get("QMC_DATA_DIR", "/content/qmc-data"),
        "secrets": secrets,
    }
    print(f"Drive: {'✅ マウント済み' if drive_ok else '⚠️ 未マウント（履歴はローカルのみ）'}")
    print(f"履歴の保存先: {info['data_dir']}")
    print(f"読み込んだ Secrets: {', '.join(secrets) or 'なし'}（HF_TOKEN 推奨）")
    return info


# ---------------------------------------------------------------------- llama.cpp
def llama_cache_dir(
    base: Path | None = None, commit: str = LLAMA_CPP_COMMIT, archs: str = CUDA_ARCHS
) -> Path:
    base = base or (APP_DRIVE_DIR / "llama-bin" if DRIVE_ROOT.exists() else Path("/content/llama-bin-cache"))
    return base / f"{commit[:10]}-sm{archs.replace(';', '_')}"


def cmake_configure_cmd(src: Path, build: Path, archs: str = CUDA_ARCHS) -> list[str]:
    return [
        "cmake", "-S", str(src), "-B", str(build), "-G", "Ninja",
        "-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
        "-DBUILD_SHARED_LIBS=OFF",  # single self-contained binary -> safe to cache/copy
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]  # fmt: skip


def install_llama_cpp(
    workdir: Path = Path("/content/llama.cpp"), local_bin: Path = Path("/content/llama-bin")
) -> Path:
    """Cell 2: get a CUDA ``llama-server`` (Drive cache if present, otherwise build ~10-20 min)."""
    local_server = local_bin / "llama-server"
    if local_server.exists():
        os.environ["QMC_LLAMA_BIN_DIR"] = str(local_bin)
        print(f"✅ llama-server: {local_server}")
        return local_server

    cache = llama_cache_dir()
    cached = cache / "llama-server"
    local_bin.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        shutil.copy2(cached, local_server)
        local_server.chmod(0o755)
        os.environ["QMC_LLAMA_BIN_DIR"] = str(local_bin)
        print(f"✅ キャッシュから llama-server を復元: {cached}")
        return local_server

    print("llama.cpp をビルドします（初回のみ。10〜20分程度）…")
    if shutil.which("ninja") is None or shutil.which("cmake") is None:
        _run(["apt-get", "install", "-y", "-qq", "cmake", "ninja-build"])
    if not (workdir / ".git").exists():
        _run(["git", "clone", "--filter=blob:none", LLAMA_CPP_REPO, str(workdir)])
    _run(["git", "-C", str(workdir), "fetch", "--depth", "1", "origin", LLAMA_CPP_COMMIT])
    _run(["git", "-C", str(workdir), "checkout", "-q", LLAMA_CPP_COMMIT])
    build = workdir / "build"
    _run(cmake_configure_cmd(workdir, build))
    _run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "-j",
            str(os.cpu_count() or 4),
        ]
    )
    built = build / "bin" / "llama-server"
    shutil.copy2(built, local_server)
    local_server.chmod(0o755)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, cached)
        print(f"💾 次回用にキャッシュしました: {cached}")
    except OSError as exc:
        print(f"⚠️ キャッシュ保存に失敗（次回もビルドが必要）: {exc}")
    os.environ["QMC_LLAMA_BIN_DIR"] = str(local_bin)
    return local_server


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------- models / launch
def prefetch_models(chat: bool = True, image: bool = True) -> None:
    """Optional: download weights up front so the first message does not wait on downloads."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    from .backends.llama_server import download_hf_file  # noqa: PLC0415
    from .config import load_config  # noqa: PLC0415

    cfg = load_config()
    if chat and not cfg.chat.remote_base_url:
        for f in (cfg.chat.model_file, cfg.chat.mmproj_file):
            print("↓", cfg.chat.hf_repo, f)
            download_hf_file(cfg.chat.hf_repo, f, cfg.hf_cache_dir)
    if image:
        print("↓", cfg.image.model_id)
        snapshot_download(cfg.image.model_id)
    print("✅ ダウンロード完了")


def launch(share: bool = False, mock: bool = False, port: int = 7860, profile: str | None = None):
    """Cell 4: build the app and start Gradio. Returns the App."""
    from .app import build_app  # noqa: PLC0415
    from .config import load_config  # noqa: PLC0415
    from .ui import build_ui  # noqa: PLC0415
    from .ui import launch as ui_launch  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(mock=mock, share=share, server_port=port)
    if profile:
        cfg.gpu_profile_override = profile
    if share:
        print("⚠️ share=True: 発行される *.gradio.live のURLは誰でもアクセスできます。")
        if not (cfg.auth_user and cfg.auth_password):
            print(
                "   Colab Secrets に QMC_AUTH_USER / QMC_AUTH_PASSWORD を設定するとログイン必須にできます。"
            )
    app = build_app(cfg)
    print(
        f"GPU: {app.gpu.name} ({app.gpu.total_gib:.0f} GiB) → {app.profile.mode} (profile={app.profile.key})"
    )
    demo = build_ui(app)
    ui_launch(app, demo, prevent_thread_lock=True, inline=False, quiet=True)
    if in_colab() and not share:
        from google.colab import output  # noqa: PLC0415

        output.serve_kernel_port_as_window(port, anchor_text="▶ Qwen Multimodal Chat を開く")
    return app
