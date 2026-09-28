"""VRAM / latency benchmark to run on a real Colab GPU:  ``python -m qmc.bench [--out docs/vram-measurements.md]``

Measures, per phase: nvidia-smi used (includes the llama-server process), torch allocated /
reserved / peak (this Python process = Qwen-Image), and wall time. Peaks are sampled by a
background thread every 0.5s. Results are appended as a Markdown section so A100 / L4 runs can
be collected in one file. Nothing here is estimated: rows are only written for phases that ran.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image  # noqa: E402

from .gpu_manager import memory_snapshot  # noqa: E402


@dataclass
class Row:
    phase: str
    seconds: float
    smi_used: float | None
    smi_peak: float | None
    torch_alloc: float | None
    torch_reserved: float | None
    torch_peak: float | None
    note: str = ""


class PeakSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak: float | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            used = memory_snapshot().nvidia_smi_used_gib
            if used is not None:
                self.peak = used if self.peak is None else max(self.peak, used)
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def measure(rows: list[Row], phase: str, fn, note: str = ""):
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    t0 = time.time()
    with PeakSampler() as sampler:
        result = fn()
    snap = memory_snapshot(phase)
    rows.append(
        Row(phase, round(time.time() - t0, 1), snap.nvidia_smi_used_gib, sampler.peak, snap.torch_allocated_gib,
            snap.torch_reserved_gib, snap.torch_max_allocated_gib, note)
    )  # fmt: skip
    print(f"[{phase}] {rows[-1]}")
    return result


def to_markdown(rows: list[Row], gpu_name: str, profile: str, extra: str = "") -> str:
    def f(v):
        return "-" if v is None else f"{v:.1f}"

    lines = [
        f"### {gpu_name} / profile `{profile}` ({dt.datetime.now().strftime('%Y-%m-%d %H:%M')})",
        "",
        extra,
        "",
        "| Phase | Time (s) | nvidia-smi used (GiB) | nvidia-smi peak (GiB) | torch alloc | torch reserved | torch peak | Note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.phase} | {r.seconds:.1f} | {f(r.smi_used)} | {f(r.smi_peak)} | {f(r.torch_alloc)} | "
            f"{f(r.torch_reserved)} | {f(r.torch_peak)} | {r.note} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/vram-measurements.md"))
    p.add_argument("--profile", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--band", type=int, default=1024)
    args = p.parse_args(argv)

    from .app import build_app
    from .backends.base import ChatParams
    from .chat_engine import ContextImage, ContextMessage
    from .config import load_config
    from .image_engine import ImageOptions

    cfg = load_config()
    if args.profile:
        cfg.gpu_profile_override = args.profile
    app = build_app(cfg)
    mm = app.manager
    chat_engine = app.controller.chat
    vision = app.controller.vision
    images = app.controller.images
    rows: list[Row] = []
    opts = ImageOptions(band=args.band, steps=args.steps, seed=1234)

    measure(rows, "baseline", lambda: None)
    measure(rows, "chat: load", lambda: mm.ensure("chat"))
    text = measure(
        rows,
        "chat: text generation",
        lambda: "".join(d.content for d in chat_engine.stream(
            [ContextMessage("user", "TerraformとPulumiの違いを200字で教えて")], ChatParams(max_tokens=400))),
    )  # fmt: skip
    img = measure(
        rows,
        "image: load",
        lambda: mm.ensure("image"),
        note="chat unloaded" if not app.profile.coresident else "",
    )
    gen = measure(rows, "image: generation", lambda: images.run(images.build_request("a futuristic data center in front of the Tokyo night skyline", opts)),
                  note=f"band {args.band}")  # fmt: skip
    gen_path = cfg.data_dir / "bench_gen.png"
    gen.save(gen_path)
    edited = measure(rows, "image: editing", lambda: images.run(images.build_edit_request(
        "make the night darker and add more neon lights", [gen], opts, parent_seeds={1234})))  # fmt: skip
    edit_path = cfg.data_dir / "bench_edit.png"
    edited.save(edit_path)
    measure(rows, "image: unload", lambda: mm.unload("image"))
    measure(
        rows,
        "vision: inference (2 images)",
        lambda: "".join(d.content for d in vision.stream(
            [ContextMessage("user", "2枚の違いを説明して")],
            [ContextImage("a", str(gen_path), "before"), ContextImage("b", str(edit_path), "after")],
            ChatParams(max_tokens=300), compare=True)),
        note="chat reloaded" if not app.profile.coresident else "",
    )  # fmt: skip
    measure(rows, "unload all", mm.unload_all)
    _ = (text, img, Image)

    md = to_markdown(
        rows, app.gpu.name, app.profile.key, extra=f"chat: {app.chat_label} / image: {app.image_label}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        fh.write("\n" + md)
    print(md)


if __name__ == "__main__":
    main()
