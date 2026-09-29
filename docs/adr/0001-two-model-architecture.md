# ADR-0001: 2モデル構成（Qwen3.8-27B + Qwen-Image-2.1）と llama-server バックエンド

- Status: Accepted
- Date: 2026-09-28

## Context

仕様では Chat(27B) / Vision(VL) / Image(Qwen-Image-2.1) の3系統が挙がっていた。
Phase 0 の調査で次が分かった（詳細: [phase0-research.md](../phase0-research.md)）。

- `ggml-org/Qwen3.8-27B-GGUF` は `image-text-to-text` で `mmproj` が同梱され、元モデルは画像・動画入力対応のネイティブVLM
- 既存の27B Notebookは Transformers ではなく **llama.cpp の `llama-server`** で L4 上の動作実績がある
- Vision Notebook (`11_qwen3_vl_image_judge.ipynb`) は Drive 上で見つからなかった

## Decision

1. **2モデル構成**にする
   - Qwen3.8-27B (GGUF Q4_K_M + mmproj。※決定当時の構成。現在の既定 Chat は Q8_K_L): Text / Vision / Reasoning
   - Qwen-Image-2.1 (diffusers): Generation / Editing
2. 27B は **llama-server をサブプロセスで起動**し、**OpenAI互換 HTTP API** で呼ぶ
3. アプリ側の Chat クライアントは「OpenAI互換エンドポイント」に対して実装し、llama-server の起動管理（ローカルプロセス）とは分離する

## Consequences

メリット
- VRAMを食うモデルが1つ減り、L4でも成立する
- 既存Notebookで実績のある llama.cpp 構成をそのまま使える（Q4_K_M, `--fit`）
- Chat の unload は「プロセス停止」なので VRAM が確実に全解放される（PyTorch のキャッシュ残りが起きない）
- OpenAI互換APIなので、将来 **vLLM / RunPod / GCP / AWS 上のサーバー**に `base_url` を変えるだけで移行できる

デメリット
- llama.cpp を Colab 上でビルドする必要がある（初回数分）。→ ビルド成果物を Google Drive にキャッシュして2回目以降は省略
- Chat の再ロードはプロセス再起動になり、L4 では Chat↔Image 切替ごとに数十秒〜1分程度かかる（ページキャッシュが効けば短縮）
- 専用VLモデルに比べて Vision の精度が劣る可能性は未評価
