# VRAM / 速度の計測結果

計測方法: Colab 上で `python -m qmc.bench` を実行すると、このファイルに結果セクションが追記されます。

- `nvidia-smi used / peak`: GPU全体（llama-server プロセス + Qwen-Image を含む）。peak は 0.5 秒間隔のサンプリング
- `torch alloc / reserved / peak`: アプリ本体の Python プロセス（= Qwen-Image-2.1）の `torch.cuda.memory_allocated()` / `memory_reserved()` / `max_memory_allocated()`
- 計測フェーズ: baseline → chat load → text generation → image load → generation → editing → image unload → vision (2 images) → unload all

```bash
cd /content/qwen-multimodal-colab
python -m qmc.bench                 # 自動判定プロファイル
python -m qmc.bench --profile l4    # プロファイル指定
```

## 計測状況

| GPU | 状態 |
| --- | --- |
| A100 40GB / 80GB | **未計測**（このリポジトリの作成環境ではColab GPUを取得していません） |
| L4 | **未計測**（同上） |

> 既存Notebookのセル出力から分かっている参考値（本アプリでの計測値ではない）:
> - L4 (22.0 GiB) で Qwen3.8-27B Q4_K_M を `--fit on --fit-target 2048 -c 8192` で起動し応答できた
> - L4 で Qwen-Image-2.1（DiT int8 / TE bf16）を全常駐させると約 21.7 GiB 使用し、VAE デコードで VRAM 不足 → `enable_model_cpu_offload()` で 1024 帯を生成できた

## 結果

（ここに `python -m qmc.bench` の結果が追記されます）
