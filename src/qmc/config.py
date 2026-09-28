"""Application configuration.

All settings have safe defaults and can be overridden with ``QMC_*`` environment variables
(Colab Secrets are exported to env vars by the notebook). No secret is ever stored in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


@dataclass
class ChatModelConfig:
    """Qwen3.8-27B served by llama.cpp ``llama-server`` (OpenAI-compatible API)."""

    hf_repo: str = "ggml-org/Qwen3.8-27B-GGUF"
    model_file: str = "Qwen3.8-27B-Q4_K_M.gguf"
    mmproj_file: str = "mmproj-Qwen3.8-27B-Q8_0.gguf"
    # When set, no local llama-server is started and this OpenAI-compatible endpoint is used
    # (e.g. vLLM on RunPod). Example: http://1.2.3.4:8000/v1
    remote_base_url: str | None = None
    remote_api_key: str | None = None
    remote_model_name: str = "qwen3.8-27b"
    host: str = "127.0.0.1"
    port: int = 8012
    ctx_size: int | None = None  # None -> taken from the GPU profile
    fit_target_mib: int | None = None  # None -> taken from the GPU profile
    image_max_tokens: int | None = None  # cap vision tokens per image (None = model default)
    startup_timeout_s: int = 900
    request_timeout_s: int = 600


@dataclass
class ImageModelConfig:
    """Qwen-Image-2.1 served by diffusers ``QwenImage21Pipeline``."""

    model_id: str = "Qwen/Qwen-Image-2.1"
    precision: str = "auto"  # auto | bf16 | int8
    default_steps: int | None = None  # None -> GPU profile
    default_band: int = 1024  # output_resolution (long side ~ this)
    max_band: int | None = None  # None -> GPU profile


@dataclass
class AppConfig:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    # Local working copy of the SQLite DB. SQLite on the Drive FUSE mount is unsafe, so the DB
    # lives on local disk and is mirrored to ``data_dir`` after every turn (see ADR-0002).
    local_db_path: Path = field(default_factory=lambda: Path("/tmp/qmc/history.db"))
    llama_bin_dir: Path | None = None
    hf_cache_dir: Path | None = None
    chat: ChatModelConfig = field(default_factory=ChatModelConfig)
    image: ImageModelConfig = field(default_factory=ImageModelConfig)
    gpu_profile_override: str | None = None  # a100_80 | a100_40 | l4 | cpu
    mock: bool = False  # CPU-only fake backends for development and UI tests
    prompt_rewrite: str = "auto"  # auto | on | off: LLM rewrites image prompts to English
    thinking_default: bool = False
    web_search: str = "auto"  # auto | on | off (default for the UI toggle)
    content_policy: str = "open"  # open | standard
    search_safesearch: str = "auto"  # auto | off | moderate | strict
    search_provider: str = "auto"  # auto | tavily | brave | duckduckgo
    search_max_results: int = 5
    max_context_messages: int = 24
    max_context_images: int = 3
    max_upload_mb: int = 30
    max_image_side: int = 2048
    server_host: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    auth_user: str | None = None
    auth_password: str | None = None
    drive_mounted: bool = False

    @property
    def db_mirror_path(self) -> Path:
        return self.data_dir / "history.db"

    def session_dir(self, session_id: str) -> Path:
        return self.data_dir / "sessions" / session_id

    def image_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "images"

    def public_dict(self) -> dict:
        """Settings safe to display / persist (no secrets)."""
        out = {}
        for f in fields(self):
            if f.name in {"auth_password"}:
                continue
            value = getattr(self, f.name)
            if f.name == "chat":
                value = {k: v for k, v in vars(value).items() if k != "remote_api_key"}
            elif f.name == "image":
                value = dict(vars(value))
            elif isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out


def default_data_dir() -> tuple[Path, bool]:
    """Return (data_dir, drive_mounted). Prefers Google Drive so history survives restarts."""
    explicit = _env("QMC_DATA_DIR")
    if explicit:
        path = Path(explicit)
        return path, str(path).startswith(str(DRIVE_ROOT)) and DRIVE_ROOT.exists()
    if DRIVE_ROOT.exists():
        return DRIVE_ROOT / "qwen-multimodal-colab" / "data", True
    if Path("/content").exists():
        return Path("/content/qmc-data"), False
    return Path("data"), False


def load_config(**overrides) -> AppConfig:
    """Build config from defaults + environment variables + keyword overrides."""
    data_dir, drive_mounted = default_data_dir()
    cfg = AppConfig(data_dir=data_dir, drive_mounted=drive_mounted)

    cfg.mock = _env_bool("QMC_MOCK", cfg.mock)
    cfg.gpu_profile_override = _env("QMC_GPU_PROFILE", cfg.gpu_profile_override)
    cfg.prompt_rewrite = _env("QMC_PROMPT_REWRITE", cfg.prompt_rewrite) or "auto"
    cfg.thinking_default = _env_bool("QMC_THINKING", cfg.thinking_default)
    cfg.web_search = _env("QMC_WEB_SEARCH", cfg.web_search) or "auto"
    cfg.content_policy = _env("QMC_CONTENT_POLICY", cfg.content_policy) or "open"
    cfg.search_safesearch = _env("QMC_SEARCH_SAFESEARCH", cfg.search_safesearch) or "auto"
    cfg.search_provider = _env("QMC_SEARCH_PROVIDER", cfg.search_provider) or "auto"
    cfg.search_max_results = _env_int("QMC_SEARCH_MAX_RESULTS", cfg.search_max_results)
    cfg.share = _env_bool("QMC_SHARE", cfg.share)
    cfg.server_port = _env_int("QMC_PORT", cfg.server_port)
    cfg.auth_user = _env("QMC_AUTH_USER")
    cfg.auth_password = _env("QMC_AUTH_PASSWORD")
    if _env("QMC_LOCAL_DB"):
        cfg.local_db_path = Path(_env("QMC_LOCAL_DB"))
    if _env("QMC_LLAMA_BIN_DIR"):
        cfg.llama_bin_dir = Path(_env("QMC_LLAMA_BIN_DIR"))
    if _env("HF_HOME"):
        cfg.hf_cache_dir = Path(_env("HF_HOME"))

    cfg.chat.hf_repo = _env("QMC_CHAT_HF_REPO", cfg.chat.hf_repo)
    cfg.chat.model_file = _env("QMC_CHAT_MODEL_FILE", cfg.chat.model_file)
    cfg.chat.mmproj_file = _env("QMC_CHAT_MMPROJ_FILE", cfg.chat.mmproj_file)
    cfg.chat.remote_base_url = _env("QMC_CHAT_BASE_URL")
    cfg.chat.remote_api_key = _env("QMC_CHAT_API_KEY")
    cfg.chat.remote_model_name = _env("QMC_CHAT_REMOTE_MODEL_NAME", cfg.chat.remote_model_name)
    if _env("QMC_CHAT_CTX"):
        cfg.chat.ctx_size = _env_int("QMC_CHAT_CTX", 0) or None
    cfg.image.precision = _env("QMC_IMAGE_PRECISION", cfg.image.precision) or "auto"

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg
