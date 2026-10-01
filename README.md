# Qwen Multimodal Colab

[English](README.md) | [日本語](README.ja.md)

A personal Gradio chat application for Google Colab with Qwen chat/vision and image generation/editing, web search, read-only GitHub research, PDFs, short videos, speech input, and optional readout. It requires an NVIDIA A100 (80GB recommended) with Qwen3.8-27B Q8_K_L; L4 and A100 40GB are not supported at the moment. That quantization change still needs GPU measurement.

## Start in Colab

1. Open the [Colab notebook](https://colab.research.google.com/github/moruku36/qwen-multimodal-colab/blob/main/Qwen-Multimodal-Colab.ipynb). **An A100 runtime is required** (A100 80GB recommended); L4 and A100 40GB are currently not supported.
2. Optionally add `HF_TOKEN` and search-provider keys through Colab Secrets.
3. Run cells 1–4 in order and open the resulting Gradio link.

Qwen3.8-27B Q8_K_L runs through llama.cpp for chat and vision with the official mmproj; Qwen-Image-2.1 runs through diffusers for image generation and editing. The implementation lives in `src/qmc/`.

## Use

Attach images, PDFs, short videos, or audio with the plus control. Use the microphone for speech input. Tools, generation settings, history, masks, and image versions are in the sidebar. GitHub URLs trigger a read-only research agent, bounded to a configurable step count; it does not write or execute repository code.

If Google Drive is mounted, history and images are stored under `MyDrive/qwen-multimodal-colab/data/`; otherwise runtime storage is temporary.

## Limits and verification

| Area | Documented default limit |
| --- | --- |
| PDF | First 6 pages; 30MB |
| Video | 30 seconds; 80MB; up to 8 frames; no video generation |
| Image editing | Up to 10 references; masks do not guarantee unchanged outside pixels |
| Speech input | Japanese small faster-whisper model loaded on CPU as needed |
| Readout | Off by default; edge-tts sends reply text to an external service |
| GPU evidence | Earlier A100 core flows used Q4_K_M; Q8_K_L and recent features still need GPU measurement |

## Local mock mode

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m qmc --mock
```

`SHARE=True` creates a public URL; configure authentication through Colab Secrets when sharing. Web search sends queries to external services. Code is MIT-licensed; model weights have separate terms, including the documented non-commercial research terms for Qwen-Image-2.1.

[Technical guide](TECHNICAL.md) · [Architecture](docs/architecture.md) · [VRAM measurements](docs/vram-measurements.md). The Japanese guide retains the full operating, troubleshooting, and improvement notes.


## Contents

- [TECHNICAL.md](TECHNICAL.md)
- [data/](data)
- [docs/](docs)
- [scripts/](scripts)
- [src/](src)
- [tests/](tests)

## Detailed documentation

The [Japanese guide](README.ja.md) retains the complete original setup instructions, configuration, examples, project status, and limitations. Supporting documents keep their existing language.

## Image generation time and Colab units

Image controls now start at the configured **1024 band / 40 steps on A100 / one image** instead of silently forcing 2048/50. The visible **legacy-quality 2048/50** preset remains available; **draft 768/20** trades detail and fidelity for less work. No GPU speedup or CU savings have been measured. Progress shows per-image elapsed time, and completion separates preparation from image processing. See the [performance review and verification limits](docs/image-performance.md). Stop/unload does not end the Colab runtime; use the session shutdown control when finished.
