"""Google Colab helpers used by the thin notebook (``Qwen-Multimodal-Colab.ipynb``).

Every function works (as a no-op or with a clear message) outside Colab, so the logic is
testable and the notebook only calls: ``setup()`` -> ``install_llama_cpp()`` -> ``launch()``.
"""

from __future__ import annotations

import builtins
import contextlib
import logging
import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# llama.cpp commit verified to have every flag we use (see docs/phase0-research.md)
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_COMMIT = "4da6337767f973e2b4d0797e5b323d77d8565e4a"
# 80 = A100, 89 = L4, 90 = H100. Only the current GPU's arch is built by default: on Colab A100
# building all three took ~70+ min (measured), one arch is ~3x faster. The Drive cache is keyed by arch.
FALLBACK_CUDA_ARCHS = "80;89"
DRIVE_ROOT = Path("/content/drive/MyDrive")
APP_DRIVE_DIR = DRIVE_ROOT / "qwen-multimodal-colab"
SECRET_NAMES = (
    "HF_TOKEN",
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "QMC_AUTH_USER",
    "QMC_AUTH_PASSWORD",
    "QMC_CHAT_API_KEY",
    "QMC_CHAT_BASE_URL",
    "QMC_IMAGE_API_KEY",
    "QMC_IMAGE_BASE_URL",
)


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
def detect_cuda_arch() -> str | None:
    """Compute capability of GPU 0 as a CMake arch string, e.g. "8.0" -> "80"."""
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except Exception:
        return None
    return parse_compute_cap(out[0]) if out else None


def parse_compute_cap(value: str) -> str | None:
    value = value.strip()
    if not value or "." not in value:
        return None
    major, minor = value.split(".", 1)
    return f"{major}{minor}" if major.isdigit() and minor.isdigit() else None


def cuda_archs() -> str:
    return os.environ.get("QMC_CUDA_ARCHS") or detect_cuda_arch() or FALLBACK_CUDA_ARCHS


def llama_cache_dir(
    base: Path | None = None, commit: str = LLAMA_CPP_COMMIT, archs: str | None = None
) -> Path:
    archs = archs or cuda_archs()
    base = base or (APP_DRIVE_DIR / "llama-bin" if DRIVE_ROOT.exists() else Path("/content/llama-bin-cache"))
    return base / f"{commit[:10]}-sm{archs.replace(';', '_')}"


def cmake_configure_cmd(src: Path, build: Path, archs: str | None = None) -> list[str]:
    archs = archs or cuda_archs()
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

    archs = cuda_archs()
    cache = llama_cache_dir(archs=archs)
    cached = cache / "llama-server"
    local_bin.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        shutil.copy2(cached, local_server)
        local_server.chmod(0o755)
        os.environ["QMC_LLAMA_BIN_DIR"] = str(local_bin)
        print(f"✅ キャッシュから llama-server を復元: {cached}")
        return local_server

    print(f"llama.cpp をビルドします（初回のみ。CUDA arch {archs}、A100で約20〜30分）…")
    if shutil.which("ninja") is None or shutil.which("cmake") is None:
        _run(["apt-get", "install", "-y", "-qq", "cmake", "ninja-build"])
    if not (workdir / ".git").exists():
        _run(["git", "clone", "--filter=blob:none", LLAMA_CPP_REPO, str(workdir)])
    _run(["git", "-C", str(workdir), "fetch", "--depth", "1", "origin", LLAMA_CPP_COMMIT])
    _run(["git", "-C", str(workdir), "checkout", "-q", LLAMA_CPP_COMMIT])
    build = workdir / "build"
    _run(cmake_configure_cmd(workdir, build, archs))
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
        print("↓", cfg.chat.hf_repo, cfg.chat.model_file)
        download_hf_file(cfg.chat.hf_repo, cfg.chat.model_file, cfg.hf_cache_dir)
        if cfg.chat.mmproj_file:
            mmproj_repo = cfg.chat.mmproj_repo or cfg.chat.hf_repo
            print("↓", mmproj_repo, cfg.chat.mmproj_file)
            download_hf_file(mmproj_repo, cfg.chat.mmproj_file, cfg.hf_cache_dir)
    if image and not cfg.image.remote_base_url:
        print("↓", cfg.image.model_id)
        snapshot_download(cfg.image.model_id)
    print("✅ ダウンロード完了")


def launch(
    share: bool = False,
    mock: bool = False,
    port: int = 7860,
    profile: str | None = None,
    web_search: str | None = None,
):
    """Cell 4: build the app and start Gradio. Returns the App."""
    from .app import build_app  # noqa: PLC0415
    from .config import load_config  # noqa: PLC0415
    from .ui import build_ui  # noqa: PLC0415
    from .ui import launch as ui_launch  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(mock=mock, share=share, server_port=port)
    if profile:
        cfg.gpu_profile_override = profile
    if web_search:
        cfg.web_search = web_search
    if share:
        print("⚠️ share=True: 発行される *.gradio.live のURLは誰でもアクセスできます。")
        if not (cfg.auth_user and cfg.auth_password):
            print(
                "   Colab Secrets に QMC_AUTH_USER / QMC_AUTH_PASSWORD を設定するとログイン必須にできます。"
            )
    # Re-running the launch cell in the same kernel: stop the previous server and free its
    # models first, otherwise port 7860 is taken ("Cannot find empty port") and VRAM is doubled.
    shutdown_previous()
    cfg.server_port = find_free_port(port)
    port = cfg.server_port
    app = build_app(cfg)
    _set_current_app(app)
    print(
        f"GPU: {app.gpu.name} ({app.gpu.total_gib:.0f} GiB) → {app.profile.mode} (profile={app.profile.key})"
    )
    demo = build_ui(app)
    # Behind the Colab port-forward proxy Gradio derives its API URL from the internal host
    # (…internal:8007) and every API call fails with 503 (verified on Colab A100). Passing the
    # public proxy URL as root_path fixes it.
    root_path = colab_proxy_url(port) if in_colab() and not share else None
    ui_launch(app, demo, prevent_thread_lock=True, inline=False, quiet=True, root_path=root_path)
    app.demo = demo
    if root_path:
        _show_link(root_path)
    return app


# The running app is kept on ``builtins`` (not a module global) so it survives the notebook
# purging/reimporting the ``qmc`` package after a ``git pull`` in the same kernel.
_REGISTRY_ATTR = "_qmc_current_app"


def _get_current_app():
    return getattr(builtins, _REGISTRY_ATTR, None)


def _set_current_app(app) -> None:
    setattr(builtins, _REGISTRY_ATTR, app)


def shutdown_previous() -> None:
    """Close the Gradio server and unload the models of a previous launch() in this kernel."""
    prev = _get_current_app()
    _set_current_app(None)
    if prev is not None:
        demo = getattr(prev, "demo", None)
        if demo is not None:
            with contextlib.suppress(Exception):
                demo.close()
        with contextlib.suppress(Exception):
            prev.manager.unload_all()
        with contextlib.suppress(Exception):
            prev.store.sync()
            prev.store.close()
        print("♻️ 前回起動したアプリを停止しました")
    with contextlib.suppress(Exception):
        import gradio as gr  # noqa: PLC0415

        gr.close_all()


def shutdown_runtime(cleanup_timeout: float = 60) -> None:
    """Best-effort app/Drive cleanup, then request Colab VM unassignment.

    Only call when the user is done or startup failed, never after a successful
    nonblocking launch. A daemon worker bounds waiting for stuck model/Drive I/O;
    timed-out or failed writes may be lost. A dead kernel cannot run this helper.
    Outside Colab, only stop the app (there is no runtime to unassign).
    """
    if cleanup_timeout <= 0:
        raise ValueError("cleanup_timeout must be positive")
    is_colab = in_colab()

    def cleanup():
        try:
            prev = _get_current_app()
            if prev is not None:
                with contextlib.suppress(Exception):
                    prev.controller.cancel()
            shutdown_previous()
        except Exception as exc:
            print(f"⚠️ アプリの停止処理に失敗しました: {exc}")
        finally:
            if is_colab:
                try:
                    from google.colab import drive  # noqa: PLC0415

                    drive.flush_and_unmount(timeout_ms=30_000)
                except Exception as exc:
                    print(f"⚠️ Drive の同期に失敗しました。未同期のデータは失われる可能性があります: {exc}")

    worker = threading.Thread(target=cleanup, name="qmc-shutdown", daemon=True)
    try:
        worker.start()
        worker.join(timeout=cleanup_timeout)
        if worker.is_alive():
            print("⚠️ 停止処理がタイムアウトしました。未保存・未同期のデータは失われる可能性があります。")
    finally:
        if is_colab:
            from google.colab import runtime  # noqa: PLC0415

            print("Colab のセッション終了を要求します。接続が切れるまでお待ちください。")
            try:
                runtime.unassign()
            except Exception:
                print("⚠️ 自動終了を確認できません。［ランタイム］→［セッションを終了］を実行してください。")
                raise
        else:
            print("Colab 外のため、ランタイムの割り当て解除は行いません。")


def show_shutdown_button() -> None:
    """Display a fresh, disarmed shutdown control; running the cell never stops the UI."""
    import ipywidgets as widgets  # noqa: PLC0415
    from IPython.display import display  # noqa: PLC0415

    confirm = widgets.Checkbox(
        value=False,
        description="生成処理が終了し、必要なファイルを保存しました（未同期データは失われる可能性があります）",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )
    button = widgets.Button(description="セッションを終了", button_style="danger", disabled=True)
    result = widgets.Output()

    def enable(change):
        button.disabled = not change["new"]

    def stop(_):
        # Check again server-side: stale/repeated click events must not disconnect twice.
        if button.disabled or not confirm.value:
            return
        button.disabled = True
        confirm.value = False
        confirm.disabled = True
        with result:
            try:
                shutdown_runtime()
            except Exception as exc:
                print(f"終了処理に失敗しました。［ランタイム］→［セッションを終了］を実行してください: {exc}")
                confirm.disabled = False

    confirm.observe(enable, names="value")
    button.on_click(stop)
    display(widgets.VBox([confirm, button, result]))


def port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(preferred: int = 7860, attempts: int = 20) -> int:
    """``preferred`` if free, otherwise the next free port (e.g. another process still holds 7860)."""
    for candidate in range(preferred, preferred + attempts):
        if port_is_free(candidate):
            if candidate != preferred:
                print(f"⚠️ ポート {preferred} は使用中のため {candidate} で起動します")
            return candidate
    raise OSError(f"ポート {preferred}-{preferred + attempts - 1} がすべて使用中です")


def colab_proxy_url(port: int) -> str | None:
    """Public https URL of a kernel port (google.colab.kernel.proxyPort)."""
    try:
        from google.colab.output import eval_js  # noqa: PLC0415

        url = eval_js(f"google.colab.kernel.proxyPort({int(port)})")
    except Exception as exc:
        print(f"⚠️ Colab のプロキシURLを取得できませんでした: {exc}")
        return None
    return str(url).rstrip("/") if url else None


def _show_link(url: str) -> None:
    try:
        from IPython.display import HTML, display  # noqa: PLC0415

        display(HTML(f'<a href="{url}/" target="_blank">▶ Qwen Multimodal Chat を開く</a>'))
    except Exception:
        pass
    print(url + "/")
