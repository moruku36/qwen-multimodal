# VRAM / 速度の計測結果

計測方法: Colab 上で `python -m qmc.bench` を実行すると、このファイルに結果セクションが追記されます。

- `nvidia-smi used / peak`: GPU全体（llama-server プロセス + Qwen-Image を含む）。peak は 0.5 秒間隔のサンプリング
- `torch alloc / reserved / peak`: アプリ本体の Python プロセス（= Qwen-Image-2.1）の `torch.cuda.memory_allocated()` / `memory_reserved()` / `max_memory_allocated()`
- 計測フェーズ: baseline → chat load → text generation → image load → generation → editing → image unload → vision (2 images) → unload all

```bash
cd /content/qwen-multimodal-colab
PYTHONPATH=src QMC_LLAMA_BIN_DIR=/content/llama-bin python -m qmc.bench                 # 自動判定プロファイル
PYTHONPATH=src QMC_LLAMA_BIN_DIR=/content/llama-bin python -m qmc.bench --profile l4    # プロファイル指定
```

## 計測状況

> **Chat モデルの変更（Q4_K_M → Q8_K_L）について**: 現在の既定 Chat は `Huihui-Qwen3.8-27B-abliterated-UD-DW-Q8_K_L.gguf`（HF 上のファイルサイズ約 27GB）です。**以下の計測結果はすべて Chat が Q4_K_M だった時点の値**で、Q8_K_L の実測ではありません。Q8_K_L の VRAM は **未計測** です。
>
> **TODO**: Q8_K_L 移行後に A100 80GB で `python -m qmc.bench` を再実行し、chat load / text generation / vision / Qwen-Image-2.1 同時常駐 / image generation / image editing の VRAM を計測して、このファイルに新しい結果セクションを追記する。

| GPU | 状態 |
| --- | --- |
| A100 80GB (A100-SXM4-80GB) | ✅ Q4_K_M で計測済み（2026-09-28, Colab Pro / High-RAM）／ **Q8_K_L は未計測** |
| A100 40GB | 未計測（今回の Colab では 80GB 版が割り当てられた） |
| L4 | 未計測 |

> 既存Notebookのセル出力から分かっている参考値（本アプリでの計測値ではない）:
> - L4 (22.0 GiB) で Qwen3.8-27B Q4_K_M を `--fit on --fit-target 2048 -c 8192` で起動し応答できた
> - L4 で Qwen-Image-2.1（DiT int8 / TE bf16）を全常駐させると約 21.7 GiB 使用し、VAE デコードで VRAM 不足 → `enable_model_cpu_offload()` で 1024 帯を生成できた

## 結果

### NVIDIA A100-SXM4-80GB / profile `a100_80` (2026-09-28) — Chat = Q4_K_M（過去の実測。Q8_K_L ではない）

chat: Qwen3.8-27B (llama.cpp GGUF Q4_K_M + mmproj, ctx 32k) / image: Qwen-Image-2.1 (DiT bf16, 全GPU常駐) / 1024×1024, 40 step

| Phase | Time (s) | nvidia-smi used (GiB) | nvidia-smi peak (GiB) | torch alloc | torch reserved | torch peak | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| chat: load | 12.6 | 20.6 | 20.6 | 0.0 | 0.0 | 0.0 | llama-server 起動（GGUF はページキャッシュ済み） |
| chat: text generation | 2.6 | 20.6 | 20.6 | 0.0 | 0.0 | 0.0 | 最大 400 token |
| image: load | 22.2 | 51.3 | 51.2 | 30.2 | 30.2 | 30.2 | chat と同時常駐 |
| image: generation | 16.0 | 58.8 | 58.8 | 30.2 | 37.6 | 36.8 | band 1024 |
| image: editing | 18.0 | 60.8 | 54.7 | 30.2 | 39.7 | 38.8 |  |
| image: unload | 22.1 | 21.2 | 60.8 | 0.0 | 0.0 | 30.2 | GPU → CPU RAM |
| vision: inference (2 images) | 12.5 | 21.2 | 21.2 | 0.0 | 0.0 | 0.0 |  |
| unload all | 2.4 | 0.6 | 21.1 | 0.0 | 0.0 | 0.0 |  |

- torch の値はアプリ本体のプロセス（= Qwen-Image）。llama-server は別プロセスなので nvidia-smi 側にだけ現れる
- Qwen-Image 単体（chat 無し）の別計測: 初回ロード 34.6s / 30.6 GiB、生成 16.4s（38.1 GiB）、編集 18.8s（40.2 GiB）
- Gradio UI 経由: 画像生成 36s（プロンプト最適化 + 初回ロード込み）、編集 18s、UI 上の VRAM 表示は同時常駐時 60.9 / 80.0 GiB

**所見（Q4_K_M 時の実測に基づく。Q8_K_L では再計測が必要）**: A100 80GB では Chat + Image を同時常駐させてもピーク約 61 GiB で、`a100_80` プロファイル（同時常駐・bf16・全GPU）の設計どおり余裕がある。
A100 40GB / L4 は未計測のため、`a100_40` / `l4` プロファイルは引き続き設計値（入れ替え方式）のまま。
