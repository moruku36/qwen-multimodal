# qwen-multimodal-colab

Google Colab の GPU をバックエンドにして、ブラウザから ChatGPT / Qwen Web 版のように使える**自分専用のマルチモーダル Qwen Web Chat**です。
1つのチャット画面で、テキストチャット・画像の理解・画像生成・画像編集・会話の継続・履歴の保存と復元ができます。

> **English summary** — A personal multimodal chat UI (Gradio) for Google Colab. Qwen3.8-27B (llama.cpp GGUF + mmproj) handles text, vision and reasoning; Qwen-Image-2.1 (diffusers) handles text-to-image and conversational image editing. It auto-detects A100 / L4 and picks a VRAM strategy, stores history in SQLite mirrored to Google Drive, and runs fully on CPU in `--mock` mode for development. **Status: verified end-to-end on Colab A100 80GB (chat, vision, generation, editing, comparison, GPU auto-detection); L4 / A100 40GB not yet verified** (see [Limitations](#15-limitations)).

![UI on Colab A100](docs/images/ui-a100-edit.jpg)

---

## 1. このプロジェクトとは

- **Chat**: Qwen3.8-27B と日本語で会話（Thinking の ON/OFF 切替可、思考過程は折りたたみ表示）
- **Vision**: 画像をアップロードして質問。生成画像・編集画像についても質問できる
- **Web検索**: 「最新の〜」「今日の〜」などは自動でWeb検索し、出典付きで回答（学習データの期限を補う）
- **Image Generation**: 「〜を描いて」「〜を生成して」で Qwen-Image-2.1 が画像生成
- **Image Editing**: 画像を添付して「背景を東京の夜景に変更して」、生成直後に「もう少し明るく」などの追加指示
- **比較**: 「元画像と今の画像の違いを説明して」→ 元画像と最新画像を両方 Vision に渡して説明
- **履歴**: SQLite + Google Drive。Colab を再起動しても過去チャットと画像が復元される
- **自動ルーティング**: モデルを意識せずに使える Auto モード＋手動切替（Chat / Vision / Generate / Edit）

## 2. Architecture

```mermaid
flowchart TD
    B[Browser] --> UI[Gradio Web UI<br/>src/qmc/ui.py]
    UI --> C[Chat Controller<br/>controller.py]
    C --> R{Intent Router<br/>router.py}
    R -->|Chat| CE[ChatEngine]
    R -->|Vision| VE[VisionEngine]
    R -->|Generate| IE[ImageEngine]
    R -->|Edit| IE
    C --> SM[SessionManager<br/>messages / images lineage]
    SM --> DB[(SQLite local)]
    DB -. snapshot each turn .-> GD[(Google Drive<br/>history.db + images)]
    CE --> MM[Model Manager<br/>lazy load / swap / OOM recovery]
    VE --> MM
    IE --> MM
    MM --> Q27[Qwen3.8-27B<br/>llama-server subprocess<br/>OpenAI-compatible API]
    MM --> QI[Qwen-Image-2.1<br/>diffusers QwenImage21Pipeline]
    MM --> GM[GPU Manager]
    GM --> A100[A100: Performance Mode]
    GM --> L4[L4: Low VRAM Mode]
```

| 層 | ファイル | 役割 |
| --- | --- | --- |
| UI | `ui.py` | Gradio。ロジックは持たず Controller のイベントを描画するだけ |
| Controller | `controller.py` | 1ターン = ルーティング → エンジン実行 → 履歴保存。UI非依存のイベントストリーム |
| Router | `router.py` | ルールベースの意図判定（日本語/英語）＋手動モード |
| Engines | `chat_engine.py`, `vision_engine.py`, `image_engine.py` | 文脈組み立て・生成パラメータ |
| Model Manager | `model_manager.py` | 遅延ロード、同時常駐可否に応じたアンロード、OOM時の回復 |
| GPU Manager | `gpu_manager.py` | GPU判定・プロファイル選択・VRAM計測 |
| Backends | `backends/llama_server.py`, `backends/qwen_image.py`, `backends/mock.py` | モデル実体。差し替え可能 |
| History | `history_manager.py`, `session_manager.py` | SQLite（ローカル）＋ Drive ミラー、画像のリビジョン系譜 |

設計判断は ADR にまとめています: [ADR-0001 2モデル構成](docs/adr/0001-two-model-architecture.md) / [ADR-0002 SQLite + Drive](docs/adr/0002-sqlite-local-copy-with-drive-mirror.md) / [ADR-0003 仕様からの変更点](docs/adr/0003-implementation-choices.md)

## 3. Supported Models

調査の詳細（既存Notebook・HF・公式GitHub・ソースコードの確認結果）は [docs/phase0-research.md](docs/phase0-research.md)。

| 用途 | モデル | 実行方法 | 量子化 | ライセンス |
| --- | --- | --- | --- | --- |
| Text / Vision / Reasoning | [`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) `Qwen3.8-27B-Q4_K_M.gguf` + `mmproj-Qwen3.8-27B-Q8_0.gguf` | llama.cpp `llama-server`（commit `4da6337` を固定ビルド） | GGUF Q4_K_M | Apache-2.0 |
| Image Generation / Editing | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) | diffusers `QwenImage21Pipeline`（main `e0abab8` 固定） | A100: bf16 / L4: DiT int8 (torchao) | Qwen Research License（非商用研究用途） |

**なぜ2モデル？** Qwen3.8-27B は画像・動画入力に対応したネイティブVLMで、GGUF に `mmproj` が同梱されています。Vision 専用モデルを常駐させる必要がないため、VRAMを節約できる2モデル構成にしました。
`11_qwen3_vl_image_judge.ipynb` は Google Drive 上で見つからず、内容を比較できていません（見つかれば `ChatBackend` を追加して3モデル構成にも拡張できます）。

## 4. A100 / L4 の違い

起動時に GPU を自動判定してプロファイルを選びます（`--profile` / `PROFILE` で上書き可）。

| | A100 80GB (`a100_80`) | A100 40GB (`a100_40`) | L4 24GB (`l4`) |
| --- | --- | --- | --- |
| 表示モード | Performance | Performance | Low VRAM |
| Chat と Image の同時常駐 | ✅ する | ❌ 入れ替え | ❌ 入れ替え |
| Qwen-Image DiT 精度 | bf16 | bf16 | int8 (torchao weight-only) |
| Qwen-Image 配置 | 全部GPU | `enable_model_cpu_offload()` | `enable_model_cpu_offload()` |
| 画像パイプラインのRAM保持 | ✅ | ✅（再有効化が速い） | ✅ |
| Chat コンテキスト長 | 32k | 32k | 16k |
| 既定ステップ数 / 最大解像度帯 | 40 / 2048 | 40 / 2048 | 28 / 1280 |
| VAE タイリング | なし | なし | 2048帯以上 |

判断根拠（計測前の設計値）: 27B Q4_K_M ≈ 19GB + KV、Qwen-Image のテキストエンコーダ（Qwen3-VL 8B, bf16）≈ 16–17GB、DiT 7B（bf16 ≈ 14GB / int8 ≈ 7GB）。
そのため 40GB 以下では同時常駐させず、`ModelManager` が必要なときだけ切り替えます（同じモデルの無駄な再ロードはしません）。
**L4 では Chat ↔ Image の切り替えごとに数十秒〜1分程度の待ち**が発生します。

> H100 など他のGPUは VRAM 量で判定します（≥70GiB → `a100_80`、≥38GiB → `a100_40`、それ以外 → `l4`）。

## 5. Quick Start

### Colab（本番）

1. `Qwen-Multimodal-Colab.ipynb` を Colab で開く（GitHub から開く or Drive にコピー）
2. ［ランタイム］→［ランタイムのタイプを変更］→ **A100**（推奨）/ **L4**
3. 🔑 Secrets に `HF_TOKEN`（推奨）を登録し、Notebook からのアクセスを許可
4. 4つのセルを上から実行 → 表示されたリンクを開く

### ローカル（開発・UI確認 / GPU不要）

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m qmc --mock        # http://localhost:7860
PYTHONPATH=src python -m pytest            # CPU テスト
```

## 6. Colab 起動方法

Notebook は4セルだけで、ロジックはすべて `src/qmc/` にあります。

| セル | 内容 |
| --- | --- |
| 1 | リポジトリを clone（2回目以降は pull） |
| 2 | `requirements-colab.txt` をインストール（**torch は入れ直さない**）、Drive マウント、`llama-server` を用意（初回は検出したGPUのCUDA archだけビルド。A100で約20〜30分 → Drive にキャッシュ、次回以降は数秒） |
| 3 | 設定（`SHARE` / `PREFETCH` / `PROFILE`） |
| 4 | `colab.launch()` で Gradio を起動し、Colab のプロキシURLを表示 |

コマンドで起動する場合:

```bash
python -m qmc                 # 自動判定
python -m qmc --profile l4    # プロファイル指定
python -m qmc --share         # 公開URL（Security 参照）
```

主な環境変数（Colab Secrets からも読み込みます）:

| 変数 | 説明 |
| --- | --- |
| `HF_TOKEN` | Hugging Face トークン（推奨） |
| `QMC_DATA_DIR` | 履歴・画像の保存先（既定: Drive の `MyDrive/qwen-multimodal-colab/data`） |
| `QMC_GPU_PROFILE` | `a100_80` / `a100_40` / `l4` |
| `QMC_AUTH_USER` / `QMC_AUTH_PASSWORD` | Gradio のログイン（`share=True` 時は必須推奨） |
| `QMC_CHAT_BASE_URL` / `QMC_CHAT_API_KEY` | 外部の OpenAI 互換サーバー（vLLM on RunPod 等）を Chat に使う |
| `QMC_PROMPT_REWRITE` | `auto` / `on` / `off` |
| `TAVILY_API_KEY` / `BRAVE_API_KEY` | Web検索プロバイダのAPIキー（任意。無ければ DuckDuckGo） |
| `QMC_WEB_SEARCH` | Web検索の既定値 `auto` / `on` / `off` |
| `QMC_SEARCH_PROVIDER` | `auto` / `tavily` / `brave` / `duckduckgo` |
| `QMC_MOCK=1` | CPU モック |

## 7. Google Drive 履歴保存

```
MyDrive/qwen-multimodal-colab/
├── data/
│   ├── history.db                 # SQLite スナップショット（毎ターン更新）
│   └── sessions/<session_id>/images/<image_id>.png
└── llama-bin/<commit>-sm80_89_90/llama-server   # ビルドキャッシュ
```

- テーブル: `sessions`, `messages`, `images`, `generations`, `model_settings`
- 画像は `parent_id` / `root_id` / `revision` で **original → revision 1 → revision 2** の系譜を保持（「元に戻して」「2個前の画像」を後から実装できる構造）
- 生成パラメータ（prompt / 最適化後prompt / steps / seed / 解像度 / 参照画像ID / 所要時間）を `generations` に保存
- ライブDBはローカル、Drive にはアトミックにスナップショット（[ADR-0002](docs/adr/0002-sqlite-local-copy-with-drive-mirror.md)）
- Drive 未マウント時は `/content/qmc-data` に保存し、UI に「ランタイム終了で消えます」と警告

## 8. Chat

- Qwen の推奨サンプリング（HF モデルカード）を使用: non-thinking `T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` / thinking `T=1.0, top_p=0.95, top_k=20`
- 「Thinking」チェックで `chat_template_kwargs.enable_thinking` を切替。思考は `reasoning_content` として分離し、折りたたみ表示
- 文脈は直近 24 メッセージ。画像は直近 3 枚だけピクセルを送り、古い画像は `[画像 #id: ...]` ラベルとして残す（トークン節約）
- ストリーミング表示、Stop / Regenerate / Clear / New Chat / 過去チャット一覧

## 9. Vision

- 画像を添付して質問 → Qwen3.8-27B（mmproj）で回答。テキスト無しで画像だけ送ると説明を返す
- 「この画像の…」「さっきの画像…」→ 直近の生成・編集画像を対象に質問
- 「元画像と今の画像の違い」→ 系譜の root と最新を、「前の画像との違い」→ 親と最新を比較
- アップロード画像は検証（形式・破損・30MB上限）、EXIF 回転補正、長辺 2048px に縮小

## 10. Image Generation

- 「〜を描いて」「〜のイラストを作って」「〜を生成して」→ Qwen-Image-2.1
- 公式推奨アスペクト比（1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16）を解像度帯に合わせて 32 の倍数で縮小
- 画像プロンプト最適化（既定 `auto`）: Chat モデルがロード済みなら、会話文脈を踏まえた英語プロンプトに変換（L4 でわざわざ再ロードはしない）
- 🎨 設定: アスペクト比、解像度帯、Steps、Seed（-1 = ランダム）
- 進捗（step 数）を表示、Stop で中断

## 11. Image Editing

- 画像添付＋「背景を〜に変更して」「〜を消して」→ 編集
- 生成・編集の直後（3ターン以内）の「もう少し明るく」「ネオンを増やして」→ 直前画像を編集
- 編集結果は親画像の revision + 1 として保存
- **編集時は系譜内で使った seed を使わない**（同じ seed・解像度だと指示を無視したほぼコピーになる既知の挙動: [diffusers#14824](https://github.com/huggingface/diffusers/issues/14824)）
- 出力のアスペクト比は元画像に合わせる。参照画像は最大10枚

### 11.5 Web検索（リアルタイム情報）

モデルの学習データには期限があるため、最新情報が必要な質問は Web 検索して出典付きで答えます。

- 画面の **🌐 Web検索**: `自動`（既定。「最新」「今日」「ニュース」「株価」「天気」「2026年」「調べて」などを含む質問だけ検索）/ `常に` / `オフ`
- 流れ: Chat モデルが検索クエリを作成 → 検索（上位5件）→ 上位3ページの本文を取得 → 番号付きの参考データとしてシステムプロンプトに入れて回答 → 末尾に **🔎 参考（Web検索）** のリンク一覧
- 検索プロバイダ（自動選択）: `TAVILY_API_KEY` があれば **Tavily**、`BRAVE_API_KEY` があれば **Brave Search**、どちらも無ければ **DuckDuckGo（`ddgs`、キー不要）**
- 検索しないときも、システムプロンプトに**現在日時（JST）**を入れ、「学習データ以降は知らない」ことをモデルに伝えています
- 検索結果は「データ」として扱い、ページ内の指示には従わないようにプロンプトで明示しています（プロンプトインジェクション対策）

## 12. GPU / VRAM

- サイドバーに GPU 名、モード（Performance / Low VRAM）、VRAM 使用量（nvidia-smi + torch）、ロード中モデルを 5 秒ごとに表示。「⏏ モデル解放」で全アンロード
- **CUDA OOM 時**: アプリは落ちず、`他モデルのunload → empty_cache → 省VRAM設定に降格 → 1回再試行` を行う
  - Qwen-Image の降格順: 全GPU常駐 → CPU offload → VAE タイリング → DiT int8
  - llama-server: コンテキスト長を半分に、`--fit-target` を増やして再起動
- 計測: `PYTHONPATH=src python -m qmc.bench` で各フェーズ（ロード直後 / テキスト生成 / Vision / 生成 / 編集 / unload後）の `memory_allocated` / `memory_reserved` / `max_memory_allocated` / nvidia-smi を [docs/vram-measurements.md](docs/vram-measurements.md) に追記

| GPU | 状態 |
| --- | --- |
| A100 80GB | ✅ 実機検証・計測済み（ピーク約 61 GiB、生成 1024² 40step 16s / 編集 18s）→ [docs/vram-measurements.md](docs/vram-measurements.md) |
| A100 40GB | 未検証・未計測 |
| L4 | **未検証・未計測** |

### Colab A100 80GB での実機検証（2026-09-28）

| MVP テスト | 結果 | メモ |
| --- | --- | --- |
| 1. 「TerraformとPulumiの違いを教えて」→ Text | ✅ | ストリーミングで日本語回答（Router: Chat） |
| 2. 画像アップロード＋「何が写っていますか？」→ Vision | ✅ | 27B + mmproj で画面内の要素まで正しく説明 |
| 3. 「東京の夜景を背景にした未来的なデータセンターを生成して」 | ✅ | 1024², 40 step, 36s（プロンプト最適化・初回ロード込み） |
| 4. 「もう少し夜を暗くして、ネオンを増やして」→ 直前画像を編集 | ✅ | 18s、rev1 として保存、seed は親と別 |
| 5. 「元画像と今の画像の違いを説明して」 | ✅ | 元画像と編集後の2枚を Vision に渡し、空の暗さ・ネオン色の違いを説明 |
| 6. ランタイム完全削除 → 再接続 → 履歴復元 | ✅ | Drive の `history.db` から会話と画像を復元（UI に「履歴を復元しました」）。llama-server も Drive キャッシュから数秒で復元 |
| 7. GPU 自動判定 | ✅ | `NVIDIA A100-SXM4-80GB (79 GiB) → Performance (profile=a100_80)` |

実機で見つけて修正した問題（PR #14）: torchao が古い（Colab 標準 0.10）、Colab のプロキシ越しに Gradio の API が 503 になる、llama.cpp の 3 アーキ同時ビルドが遅すぎる。

| 生成 | 編集（Chat + Image 同時常駐, 60.9/80 GiB） | ランタイム再作成後の履歴復元 |
| --- | --- | --- |
| ![generate](docs/images/ui-a100-generate.jpg) | ![edit](docs/images/ui-a100-edit.jpg) | ![restore](docs/images/ui-a100-restored.jpg) |

## 13. Troubleshooting

| 症状 | 対処 |
| --- | --- |
| 画面は出るが「Connection to the server was lost」 | Colab のポート転送越しに Gradio の API URL がずれるのが原因。`colab.launch()` は `google.colab.kernel.proxyPort()` を `root_path` に渡して解決済み（自前で `demo.launch()` する場合も同様に指定） |
| `cannot import name 'FqnToConfig' from 'torchao.quantization'` | Colab 標準の torchao 0.10 が古い。`pip install -U "torchao>=0.13"`（`requirements-colab.txt` で指定済み） |
| `llama-server が見つかりません` | Cell 2 を実行（`colab.install_llama_cpp()`）。ビルド失敗時は `/content/llama.cpp` を削除して再実行 |
| `Hugging Face からのダウンロードに失敗` | `HF_TOKEN` を Secrets に登録、ディスク空き（約70GB）を確認、再実行（3回リトライ済み） |
| `QwenImage21Pipeline` が import できない | `requirements-colab.txt` の diffusers（GitHub main 固定）が入っているか確認。PyPI の 0.40.0 には未収録 |
| 画像生成で `CUDA OOM` | 解像度帯・Steps を下げる。L4 は 1024 帯推奨。自動回復後も失敗する場合は「⏏ モデル解放」 |
| L4 で返答が遅い | Chat ↔ Image のモデル切替中（ステータス表示を確認）。連続して画像を作ると切替が減る |
| 編集結果が元画像とほぼ同じ | seed を変える / 指示を具体的にする / 解像度帯を 768 にする |
| 履歴が消えた | Drive マウントを確認（UI の「履歴」表示）。DB 破損時は自動退避・復元し、UI に通知 |
| Colab が切断された | 再接続して Cell 1〜4 を再実行。最後に完了したターンまでは Drive から復元される |
| リンクが開けない | Cell 4 を再実行、または `SHARE=True`（Security 参照） |

## 14. Security

- **`share=True` を使うと `*.gradio.live` の公開URLが発行され、URLを知っている人は誰でもアクセスできます**（あなたの GPU・会話履歴・画像を含む）。使う場合は `QMC_AUTH_USER` / `QMC_AUTH_PASSWORD` を Colab Secrets に設定してログイン必須にしてください。既定は `share=False`（Colab のポート転送URLで開く。gradio.live の公開URLは作らない）
- Hugging Face / GitHub トークン、API キー、Drive の認証情報はコードやNotebookに書かず、**Colab Secrets / 環境変数**を使います
- `llama-server` は `127.0.0.1` にのみバインドし、起動ごとにランダムな API キーを付与
- `.gitignore` でモデル・生成画像・DB・`.env`・認証情報ファイルを除外
- UI の設定表示・ログにはパスワード / APIキーを出しません（`AppConfig.public_dict()`）
- Web検索を使うと、**検索クエリが外部の検索サービス**（Tavily / Brave / DuckDuckGo 等）に送信され、参考ページにもアクセスします。送りたくない会話では 🌐 Web検索 を `オフ` にしてください

## 15. Limitations

- **実機検証は Colab A100 80GB のみ**（2026-09-28）。Gradio UI 上で MVP テスト 1〜5（Chat / Vision / 生成 / 直前画像の編集 / 元画像と現在の比較）と Test 7（A100 → Performance 自動選択）を確認し、`python -m qmc.bench` で VRAM を実測した。**L4 と A100 40GB は未検証**（`l4` / `a100_40` プロファイルは設計値）
- CPU 上のユニットテスト（117件）とモックバックエンドでの Gradio E2E（Playwright）も通過
- VRAM の実測値は A100 80GB のみ。A100 40GB / L4 のプロファイル値は設計見積もり
- Intent Router はルールベースのため誤判定があり得る（手動モードで上書き可能）
- シングルユーザー前提（GPU処理は1件ずつ直列）
- Qwen-Image-2.1 は Qwen Research License（非商用研究用途）
- 「元に戻して」「2個前の画像を使って」は、データ構造（系譜）は実装済みだが、会話コマンドとしては未実装

## 16. Roadmap

- [x] Colab A100 80GB での実機検証と VRAM 実測（`docs/vram-measurements.md`）
- [ ] Colab L4 / A100 40GB での実機検証と VRAM 実測
- [ ] 「元に戻して」「2個前の画像」などの系譜コマンド
- [ ] バックエンドの外部化: RunPod / Vast.ai / GCP / AWS 上の vLLM（`QMC_CHAT_BASE_URL` で Chat は対応済み）、画像側の HTTP バックエンド
- [ ] FastAPI バックエンド + 別フロントエンド / PWA（Controller は UI 非依存）
- [ ] Docker 化
- [ ] MTP（`mtp-Qwen3.8-27B-*.gguf`）による投機的デコード高速化
- [x] Web検索（Tavily / Brave / DuckDuckGo）
- [ ] RAG（手元ドキュメント）/ 音声入出力（STT / TTS）
- [ ] マルチユーザー（認証・DBの分離）

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
PYTHONPATH=src python -m pytest                 # CPU tests
PYTHONPATH=src python -m pytest -m gpu tests/integration   # Colab GPU only
```

## License

MIT（ソースコードのみ）。モデルの重みはそれぞれのライセンスに従います。
