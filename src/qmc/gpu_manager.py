"""GPU detection, GPU profile selection and VRAM measurement.

Everything that decides *what* to do is a pure function (testable on CPU). Only
``detect_gpu`` / ``memory_snapshot`` touch torch / nvidia-smi, and both degrade gracefully.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass

log = logging.getLogger(__name__)

GIB = 1024**3


@dataclass(frozen=True)
class GPUInfo:
    name: str
    total_gib: float
    bf16: bool = True
    compute_capability: str | None = None

    @property
    def available(self) -> bool:
        return self.total_gib > 0


NO_GPU = GPUInfo(name="CPU (no CUDA GPU)", total_gib=0.0, bf16=False)


@dataclass(frozen=True)
class GPUProfile:
    """How to use a given GPU. Values are design defaults; tune with docs/vram-measurements.md."""

    key: str
    mode: str  # "Performance" | "Low VRAM" | "CPU (mock)"
    # Can the chat model (llama-server) and the image pipeline be on the GPU at the same time?
    coresident: bool
    image_precision: str  # bf16 | int8 (DiT weights)
    image_placement: str  # "gpu" (all resident) | "model_offload" (enable_model_cpu_offload)
    keep_image_in_ram: bool  # keep the unloaded image pipeline in CPU RAM for fast re-activation
    chat_ctx_size: int
    chat_fit_target_mib: int
    image_default_steps: int
    image_max_band: int
    vae_tiling_band: int  # enable VAE tiling at/above this band (0 = never)

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, GPUProfile] = {
    # A100 80GB (main target): 27B Q8_K_L + Qwen-Image bf16 fully resident. The Q4_K_M numbers in
    # docs/vram-measurements.md (~22GB chat) are historical; Q8_K_L is not yet measured (re-run qmc.bench).
    "a100_80": GPUProfile(
        key="a100_80",
        mode="Performance",
        coresident=True,
        image_precision="bf16",
        image_placement="gpu",
        keep_image_in_ram=True,
        chat_ctx_size=32768,
        chat_fit_target_mib=2048,
        image_default_steps=40,
        image_max_band=2048,
        vae_tiling_band=0,
    ),
    # A100 40GB: chat (~22GB) + text encoder (~17GB) would exceed 40GB, so chat and image swap,
    # but the image pipeline stays in CPU RAM (Colab A100 has ~83GB RAM) and runs in bf16.
    "a100_40": GPUProfile(
        key="a100_40",
        mode="Performance",
        coresident=False,
        image_precision="bf16",
        image_placement="model_offload",
        keep_image_in_ram=True,
        chat_ctx_size=32768,
        chat_fit_target_mib=2048,
        image_default_steps=40,
        image_max_band=2048,
        vae_tiling_band=0,
    ),
    # L4 24GB (22 GiB usable): proven settings from the existing notebooks.
    "l4": GPUProfile(
        key="l4",
        mode="Low VRAM",
        coresident=False,
        image_precision="int8",
        image_placement="model_offload",
        keep_image_in_ram=True,
        chat_ctx_size=16384,
        chat_fit_target_mib=2048,
        image_default_steps=28,
        image_max_band=1280,
        vae_tiling_band=2048,
    ),
    "cpu": GPUProfile(
        key="cpu",
        mode="CPU (mock)",
        coresident=True,
        image_precision="bf16",
        image_placement="gpu",
        keep_image_in_ram=True,
        chat_ctx_size=8192,
        chat_fit_target_mib=1024,
        image_default_steps=4,
        image_max_band=1024,
        vae_tiling_band=0,
    ),
}


def select_profile(gpu: GPUInfo, override: str | None = None) -> GPUProfile:
    """Pick a profile from the GPU name / VRAM. ``override`` wins when valid."""
    if override:
        key = override.strip().lower()
        if key in PROFILES:
            return PROFILES[key]
        log.warning("Unknown GPU profile override %r, falling back to auto-detection", override)

    if not gpu.available:
        return PROFILES["cpu"]

    name = gpu.name.upper()
    total = gpu.total_gib
    if "A100" in name:
        return PROFILES["a100_80"] if total >= 70 else PROFILES["a100_40"]
    if "L4" in name.replace("-", " ").split():
        return PROFILES["l4"]
    # Unknown GPU (H100, T4, ...): decide by VRAM only.
    if total >= 70:
        return PROFILES["a100_80"]
    if total >= 38:
        return PROFILES["a100_40"]
    if total >= 20:
        return PROFILES["l4"]
    log.warning("GPU %s has only %.1f GiB; the 27B model will not fit well. Using Low VRAM.", gpu.name, total)
    return PROFILES["l4"]


def detect_gpu() -> GPUInfo:
    """Detect the first CUDA GPU via torch, falling back to nvidia-smi, else NO_GPU."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GPUInfo(
                name=props.name,
                total_gib=round(props.total_memory / GIB, 1),
                bf16=bool(torch.cuda.is_bf16_supported()),
                compute_capability=f"{props.major}.{props.minor}",
            )
    except Exception as exc:  # torch missing or broken CUDA
        log.debug("torch GPU detection failed: %s", exc)

    smi = _nvidia_smi(["name", "memory.total"])
    if smi:
        name, total_mib = smi[0]
        try:
            return GPUInfo(name=name, total_gib=round(float(total_mib) / 1024, 1))
        except ValueError:
            pass
    return NO_GPU


def _nvidia_smi(fields: list[str]) -> list[list[str]]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:
        return []
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


@dataclass
class MemorySnapshot:
    """VRAM usage. ``nvidia_smi_used_gib`` includes other processes (e.g. llama-server)."""

    label: str = ""
    torch_allocated_gib: float | None = None
    torch_reserved_gib: float | None = None
    torch_max_allocated_gib: float | None = None
    nvidia_smi_used_gib: float | None = None
    nvidia_smi_total_gib: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = []
        if self.nvidia_smi_used_gib is not None and self.nvidia_smi_total_gib:
            parts.append(f"GPU {self.nvidia_smi_used_gib:.1f}/{self.nvidia_smi_total_gib:.1f} GiB")
        if self.torch_allocated_gib is not None:
            parts.append(
                f"torch alloc {self.torch_allocated_gib:.1f} / reserved {self.torch_reserved_gib:.1f} GiB"
            )
        return " | ".join(parts) or "VRAM: n/a"


def memory_snapshot(label: str = "") -> MemorySnapshot:
    snap = MemorySnapshot(label=label)
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            snap.torch_allocated_gib = round(torch.cuda.memory_allocated() / GIB, 2)
            snap.torch_reserved_gib = round(torch.cuda.memory_reserved() / GIB, 2)
            snap.torch_max_allocated_gib = round(torch.cuda.max_memory_allocated() / GIB, 2)
    except Exception:
        pass
    smi = _nvidia_smi(["memory.used", "memory.total"])
    if smi:
        try:
            snap.nvidia_smi_used_gib = round(float(smi[0][0]) / 1024, 2)
            snap.nvidia_smi_total_gib = round(float(smi[0][1]) / 1024, 2)
        except ValueError:
            pass
    return snap


def free_cuda_memory() -> None:
    """gc + empty_cache; safe to call without torch/CUDA."""
    import gc

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def is_oom_error(exc: BaseException) -> bool:
    """True for CUDA out-of-memory errors (works without importing torch)."""
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
        return True
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cublas" in text or "tried to allocate" in text)
