# Qwen Multimodal Colab

Google Colab で動く個人用の AI チャット画面です。**会話・画像の理解と生成・PDFや短い動画の読解・音声入力**を一つの画面で使えます。会話と画像は Google Drive に保存できます。

> **English:** A personal Gradio chat app for Google Colab. Qwen3.8-27B handles chat and vision; Qwen-Image-2.1 handles image creation and edits. It also supports search, PDFs, short videos, speech input and optional readout. Earlier core flows were tested on an A100; recent features have CPU tests and still need Colab GPU verification.

## まず使う

1. [Colab Notebook を開く](https://colab.research.google.com/github/moruku36/qwen-multimodal-colab/blob/main/Qwen-Multimodal-Colab.ipynb)。ランタイムは **A100 推奨**、L4 も選べます。
2. Colab の「🔑 Secrets」に `HF_TOKEN` を登録します（任意ですが推奨）。検索用の `TAVILY_API_KEY` / `BRAVE_API_KEY` も任意です。
3. Notebook の **Cell 1 → 4** を順番に実行し、最後に表示されるリンクから画面を開きます。初回はモデルのダウンロードに時間がかかります。

Notebook は起動用です。機能本体は [`src/qmc/`](src/qmc/) にあります。Colab 以外で画面を試す場合は、下の「開発・モックモード」を参照してください。

## 画面の使い方

画面の中央に質問を書いて送信します。**＋**から画像・PDF・動画を添付できます。左側にはチャット履歴が約7件見える高さで並びます。細かい設定や画像の系譜は必要なときだけ開けます。

| したいこと | 操作 |
| --- | --- |
| 普通に質問する | 入力欄へ書いて送信。既定では Web 検索も行います |
| 画像を作る | 「猫のイラストを描いて」のように頼む |
| 画像を編集する | 画像を添付して「背景を夜景にして」。複数枚の参照も可能 |
| 生成した4枚から選ぶ | 画像設定で「4枚」を選ぶ → 左の「画像の系譜・バリエーション」で `var1`〜`var4` を選択 → 編集を指示 |
| PDF・動画を読む | ファイルを添付して「要約して」「何が映ってる？」と聞く |
| 声で入力する | 「マイク・音声ファイル」で録音・選択し、大きな **音声を送信** ボタンを押す |
| 回答を音声で聞く | **回答を読み上げる** をオンにして送信 |
| 前の会話を開く | 左の「チャット履歴」から選ぶ。新規作成・削除も左側で操作 |

### 画像の編集と復元

左の「画像の系譜・バリエーション」を開き、サムネイルを選ぶと編集対象になります。`元に戻す` は元画像の画素を新しい版として保存し、画像モデルを再実行しません。`マスクで編集` を開いて範囲を塗ることもできます。ただしローカル画像モデルのマスクは目安であり、範囲外の画素を完全に固定する機能ではありません。

### どこに保存される？

Drive をマウントした場合、履歴・生成画像は `MyDrive/qwen-multimodal-colab/data/` に保存されます。Drive が使えない場合はランタイム内に保存され、ランタイム終了時に消えます。画面左の「システム状態」で保存先を確認できます。会話が長くなっても、モデルへ送る文脈量だけを制限し、保存した履歴は消しません。

## 主な制限

| 項目 | 制限・補足 |
| --- | --- |
| PDF | 既定で先頭6ページ、30 MBまで |
| 動画 | 30秒・80 MBまで。最大8フレームを読解。動画生成は非対応 |
| 画像の参照 | 編集で最大10枚。4枚の生成は順番に処理 |
| 音声入力 | `faster-whisper` の日本語 `small` を CPU で必要時に読み込み |
| 読み上げ | 既定オフ。`edge-tts` が外部サービスへ回答本文を送信するため、ネットワークが必要 |
| GPU | 旧版の基本機能は A100 80GB で確認済み。新機能、L4、A100 40GB は実機検証が必要 |

## 困ったとき

| 症状 | 確認すること |
| --- | --- |
| 画面が開かない | Colab の Cell 4 を再実行。切断時は Cell 1 から再実行 |
| 初回ロードが進まない | `HF_TOKEN`、Colab のディスク空き、ランタイムの GPU を確認 |
| 画像生成でメモリ不足 | 画像設定の解像度帯とバリエーションを下げ、必要なら「システム状態」からモデルを解放 |
| PDF・動画が読めない | PDF は30 MB以内、動画は30秒・80 MB以内。動画には `ffmpeg` が必要 |
| 音声入力・読み上げが動かない | Cell 2 を再実行。`faster-whisper` / `edge-tts` とネットワークを確認 |
| 履歴が見えない | Drive のマウントと左の「システム状態」を確認 |

詳しいエラー対処、GPU の測定値、モデル構成は [技術ガイド](TECHNICAL.md) にまとめています。

## 設定と安全性

既定は **Web 検索「常に」**、コンテンツ方針「開放」です。「詳細設定」で検索を「自動」や「オフ」に変更できます。検索すると質問に応じたクエリが外部サービスへ送信されます。未成年者の性的内容は扱いません。

| 環境変数 | 用途 |
| --- | --- |
| `QMC_DATA_DIR` | 履歴と画像の保存先 |
| `QMC_WEB_SEARCH` | `on`（既定）/ `auto` / `off` |
| `QMC_CHAT_BASE_URL` / `QMC_CHAT_API_KEY` | 外部の Chat API を使う |
| `QMC_IMAGE_BASE_URL` / `QMC_IMAGE_API_KEY` | 外部の画像 API を使う。形式は [Image HTTP API](docs/image-http-api.md) |
| `QMC_TTS` / `QMC_TTS_VOICE` | 読み上げの初期設定と声 |
| `QMC_AUTH_USER` / `QMC_AUTH_PASSWORD` | 共有 URL にログインを付ける |

`SHARE=True` で発行される URL は公開されます。共有する場合は認証情報を Colab Secrets に設定してください。API キーやパスワードを Notebook に直接書かないでください。全設定は [技術ガイド](TECHNICAL.md) を参照してください。

## 開発・モックモード

GPU を使わず画面と機能の流れを確認できます。

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python -m qmc --mock
```

テストは `PYTHONPATH=src python -m pytest`。外部画像 API の開発用モックは `python scripts/image_http_stub.py --port 8013` で起動できます。設計の根拠は [`docs/adr/`](docs/adr/) にあります。

ソースコードは MIT License。モデルの重みは各モデルのライセンスに従います。Qwen-Image-2.1 は非商用研究用途のライセンスです。
