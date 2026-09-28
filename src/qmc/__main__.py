"""CLI entry point: ``python -m qmc [--mock] [--share] [--port 7860] [--profile l4]``."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Must be set before torch initializes CUDA (reduces fragmentation; from the Qwen-Image notebook).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="qmc", description="Qwen multimodal web chat")
    p.add_argument("--mock", action="store_true", help="CPU-only fake models (UI / dev)")
    p.add_argument("--share", action="store_true", help="Gradio public share link (exposed to the internet!)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--profile", choices=["a100_80", "a100_40", "l4", "cpu"], default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--no-launch", action="store_true", help="build everything but do not start the server")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from .app import build_app
    from .config import load_config
    from .ui import build_ui, launch

    cfg = load_config()
    if args.mock:
        cfg.mock = True
    if args.share:
        cfg.share = True
    if args.port:
        cfg.server_port = args.port
    if args.profile:
        cfg.gpu_profile_override = args.profile
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if cfg.share and not (cfg.auth_user and cfg.auth_password):
        logging.warning(
            "share=True without QMC_AUTH_USER/QMC_AUTH_PASSWORD: anyone with the URL can use your GPU and history!"
        )
    app = build_app(cfg)
    demo = build_ui(app)
    if not args.no_launch:
        launch(app, demo)


if __name__ == "__main__":
    main()
