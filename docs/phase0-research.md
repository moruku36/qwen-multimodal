# Phase 0: 既存Notebook調査・モデル仕様確認

調査日: 2026-09-28
方針: **推測で実装しない**。既存Notebook（Google Drive）、Hugging Face、公式GitHub、ライブラリのソースコードを一次情報とした。

## 1. 既存Notebook

### 1.1 `qwen38_27b_colab_fixed.ipynb`（Google Drive）

| 項目 | 内容 |
| --- | --- |
| 推論エンジン | **llama.cpp**（`ggml-org/llama.cpp` をCUDA有効で `cmake` ビルド。`-DGGML_CUDA=ON`） |
| モデル | `ggml-org/Qwen3.8-27B-GGUF` / `Qwen3.8-27B-Q4_K_M.gguf`（約19 GB） |
| 量子化 | GGUF **Q4_K_M** |
| 起動 | `llama-server -m <gguf> --fit on --fit-target 2048 -c 8192 --jinja --host 127.0.0.1 --port 8012` |
| GPU配置 | `-ngl` 未指定。`--fit on` で空きVRAMに合わせて自動調整 |
| 実行確認済みGPU | **L4（22.0 GiB）**（セル出力の `nvidia-smi` より） |
| 出力 | `<think>...</think>` の思考ブロック付きで日本語応答（thinkingがデフォルトON） |
| その他 | `transformers>=5,<6` / `huggingface_hub` / `accelerate` を入れているが、推論自体はllama.cppのみ。HF側セルは import エラーが出ていた |

> 注: 同Notebookには本アプリと無関係な実験セル（モデルの拒否挙動を除去する "abliteration" コード）が含まれていた。本プロジェクトには移植していない。

### 1.2 `Qwen-Image-2.1-Colab-Pro.ipynb`（Google Drive）

| 項目 | 内容 |
| --- | --- |
| ライブラリ | `diffusers`（**GitHub main**）, `transformers>=5.17,<5.18`, `accelerate`, `safetensors`, `pillow`, `torchao` |
| モデル | `Qwen/Qwen-Image-2.1` / `QwenImage21Pipeline` + `AutoencoderKLQwenImage21`（VAEはbf16で明示ロード） |
| 精度 | VRAM ≥ 38 GiB → bf16、未満（L4）→ **DiTのみ torchao int8 weight-only**（`Int8WeightOnlyConfig` + `PipelineQuantizationConfig`）。テキストエンコーダは bf16 のまま（int8化するとオフロードが一部しか効かずエラーになるため、とNotebookに記載） |
| 配置 | VRAM ≥ 70 GiB → 全部GPU常駐、未満 → `enable_model_cpu_offload()` |
| 実測メモ | 「L4 の int8 で全部常駐すると約 21.7 GiB 使用し、VAEデコードでVRAM不足」（Notebook内コメント） |
| VAE | 2048帯 & VRAM < 38 GiB のとき `vae.enable_tiling()` |
| OOM処理 | `torch.OutOfMemoryError` を捕捉 → `empty_cache` → offload/tiling 有効化 → 1回だけ再試行 |
| 環境変数 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（CUDA初期化前に設定） |
| 推奨 | 下書き 1024帯/25 step/CFG 1 → 本番 同seed 40 step。`true_cfg_scale` はネガティブ使用時のみ ~2 |
| 実行確認済みGPU | **L4（22.0 GiB, BF16対応）** |

### 1.3 `11_qwen3_vl_image_judge.ipynb`

**Google Drive 上で見つからなかった**（タイトル検索・全 `.ipynb` 一覧の両方で確認）。
そのため中身は確認できていない。下記の通り 27B モデル自体がVision対応であることを一次情報で確認できたため、2モデル構成を採用した（[ADR-0001](adr/0001-two-model-architecture.md)）。
Vision専用Notebookに明確なメリット（例: 判定タスクで精度が高い）がある場合は、`ChatBackend` を追加実装すれば3モデル構成へ拡張できる。

## 2. モデル仕様（一次情報）

### 2.1 Qwen3.8-27B

- `https://huggingface.co/api/models/ggml-org/Qwen3.8-27B-GGUF`
  - `pipeline_tag: image-text-to-text`、`base_model: Qwen/Qwen3.8-27B`、license: apache-2.0
  - ファイル: `Qwen3.8-27B-{BF16,Q4_K_M,Q8_0}.gguf`, **`mmproj-Qwen3.8-27B-{BF16,Q8_0}.gguf`**, `mtp-Qwen3.8-27B-*.gguf`
- `https://huggingface.co/Qwen/Qwen3.8-27B`
  - 「native vision-language model that understands images and videos」→ **テキスト・画像・動画入力に対応**
  - Thinking はデフォルトON。リクエスト単位で `chat_template_kwargs: {"enable_thinking": false}` で無効化可能
  - 推奨サンプリング: thinking: `temperature=1.0, top_p=0.95, top_k=20, min_p=0`、non-thinking: `temperature=0.7, top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5`
  - Context 262,144（ネイティブ）

**結論: Vision は 27B + mmproj で処理できる → Vision専用モデルを常駐させる必要はない。**

### 2.2 llama.cpp `llama-server`（commit `4da6337`）

`common/arg.cpp` と `tools/server/` のソースで確認:

- `-mm/--mmproj FILE`（Vision projector）, `--no-mmproj`, `--mmproj-offload`
- `--fit [on|off]`（デフォルトon）, `-fitt/--fit-target MiB`（デフォルト1024）
- `-c/--ctx-size`, `-np/--parallel`, `-fa/--flash-attn`, `--jinja`, `--api-key`, `--no-webui`, `-cram/--cache-ram`, `--image-max-tokens`
- `--reasoning-format deepseek` → 思考を `reasoning_content` に分離
- `/v1/chat/completions`: `image_url` に `data:image/...;base64,...` を受け付ける（`server-common.cpp`）
- リクエストの `chat_template_kwargs.enable_thinking`（bool）を解釈（`server-common.cpp` L1339付近）
- `/health` は API キー不要（public）

### 2.3 Qwen-Image-2.1

- `https://huggingface.co/Qwen/Qwen-Image-2.1`, `https://github.com/QwenLM/Qwen-Image-2.1`
  - Text-to-Image / **Image Editing** / RGBA透過生成 / 最大10枚の参照画像
  - DiT 7B（32層, block-causal）, テキストエンコーダ **Qwen3-VL 8B**, 64ch RGBA VAE（16×圧縮）
  - 推奨解像度（2048帯）: 1:1 2048², 4:3 2400×1792, 3:4, 3:2 2528×1696, 2:3, 16:9 2752×1536, 9:16
  - License: **Qwen Research License Agreement**
- diffusers `QwenImage21Pipeline`（`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`、commit `e0abab8`、version `0.41.0.dev0`）
  - `__call__(prompt, image=None, negative_prompt=None, true_cfg_scale=1.0, height=None, width=None, num_inference_steps=40, generator=None, callback_on_step_end=None, output_resolution=1024, use_kv_cache=True, ...)`
  - 編集: `image=` に PIL画像（またはリスト）を渡す。`width/height` 省略時は最後の条件画像のアスペクト比 × `output_resolution` から自動計算
  - `model_cpu_offload_seq = "text_encoder->transformer->vae"`
  - PyPI の `diffusers 0.40.0` には **未収録**（wheel内に `qwenimage21` なし）→ GitHub main をコミット固定で使う
- 既知の注意点: [diffusers#14824](https://github.com/huggingface/diffusers/issues/14824)
  - 生成と編集で**同じseed・同じ解像度**を使うと、編集指示が無視された「ハロー付きのほぼコピー」になる
  - → 本アプリでは **編集時は必ず親画像と異なる seed を使う**

## 3. GPU / VRAM 見積もり（計測前の設計値）

| コンポーネント | 見積もり | 根拠 |
| --- | --- | --- |
| Qwen3.8-27B Q4_K_M | 重み ~19 GB + KV/mmproj 数GB | HF ファイルサイズ |
| Qwen-Image テキストエンコーダ (Qwen3-VL 8B, bf16) | ~16–17 GB | 8B × 2 bytes |
| Qwen-Image DiT 7B | bf16 ~14 GB / int8 ~7 GB | 7B × 2 / 1 bytes |
| Qwen-Image 全常駐 (L4, int8) | ~21.7 GiB（VAEデコードでOOM） | 既存Notebookコメント |

→ **L4 (22 GiB) と A100 40GB では Chat と Image の同時常駐は不可**、A100 80GB なら同時常駐可能、と判断した。
実測値は `python -m qmc.bench` で取得し `docs/vram-measurements.md` に記録する（未計測のGPUは「未計測」と明記）。
