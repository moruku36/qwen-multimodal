# アーキテクチャ図

現在の構成（**A100 80GB + Qwen3.8-27B Q8_K_L + Qwen-Image-2.1**）を図にまとめています。図は GitHub 上で Mermaid として表示されます。VRAM の数値は載せていません（Q8_K_L は未計測。[`vram-measurements.md`](vram-measurements.md) 参照）。

- [1. システム全体](#1-システム全体)
- [2. 1ターンの処理の流れ](#2-1ターンの処理の流れ)
- [3. 画像生成の流れ（固有キャラの外見検索つき）](#3-画像生成の流れ固有キャラの外見検索つき)
- [4. チャットの流れ（検索・調査エージェント）](#4-チャットの流れ検索調査エージェント)
- [5. モデルと GPU の配置](#5-モデルと-gpu-の配置)
- [6. データの保存](#6-データの保存)
- [7. モジュール構成](#7-モジュール構成)

## 1. システム全体

Colab の1つの GPU ランタイムの中で、Gradio 画面・アプリ本体・2つのモデルが動きます。Chat / Vision は別プロセスの `llama-server`、画像は同じ Python プロセスの diffusers です。

```mermaid
flowchart LR
    User(["ユーザー<br/>ブラウザ"])

    subgraph Colab["Google Colab ランタイム (A100 80GB)"]
        UI["Gradio Web UI<br/>ui.py"]
        subgraph App["アプリ本体 src/qmc"]
            CTRL["Chat Controller<br/>controller.py"]
            ROUTER{"Intent Router<br/>router.py"}
            ENG["Chat / Vision / Image<br/>Engines"]
            AGENT["調査エージェント<br/>agent.py"]
            SEARCH["Web検索エンジン<br/>search_engine.py"]
            MM["Model Manager<br/>遅延ロード / 入替 / OOM回復"]
        end
        subgraph Models["モデル"]
            LLAMA["llama-server (別プロセス)<br/>Qwen3.8-27B Q8_K_L<br/>+ mmproj Q8_0<br/>OpenAI互換API"]
            QIMG["Qwen-Image-2.1<br/>diffusers QwenImage21Pipeline"]
        end
        DB[("SQLite<br/>ローカル")]
        ASR["faster-whisper<br/>(CPU / 音声入力)"]
    end

    Drive[("Google Drive<br/>history.db + 画像")]
    HF[["Hugging Face<br/>モデル取得"]]
    Web[["Web検索<br/>Brave / DuckDuckGo / Tavily"]]
    GH[["GitHub<br/>リポジトリ読み取り"]]
    TTS[["edge-tts<br/>読み上げ (任意)"]]

    User <--> UI
    UI <--> CTRL
    CTRL --> ROUTER
    ROUTER --> ENG
    CTRL --> AGENT
    CTRL --> SEARCH
    AGENT --> SEARCH
    AGENT --> GH
    SEARCH --> Web
    ENG --> MM
    MM --> LLAMA
    MM --> QIMG
    CTRL --> DB
    DB -. "ターンごとにスナップショット" .-> Drive
    HF -. "初回ダウンロード" .-> LLAMA
    HF -. "初回ダウンロード" .-> QIMG
    UI --> ASR
    CTRL -. "任意" .-> TTS
```

外部の Chat / 画像サーバーを使うこともできます（`QMC_CHAT_BASE_URL` / `QMC_IMAGE_BASE_URL`）。その場合 `MM` の先は `llama-server` / diffusers ではなく、OpenAI 互換 API と [Image HTTP API](image-http-api.md) になります。

## 2. 1ターンの処理の流れ

ユーザーの発言は必ず Router を通ります。Router はルールベースで、判定の理由は画面に表示されます。手動モードで上書きもできます。

```mermaid
flowchart TD
    IN["ユーザー入力<br/>テキスト + 添付 (画像 / PDF / 動画 / 音声)"] --> PRE["前処理<br/>PDF→ページ画像 / 動画→フレーム / 音声→文字起こし"]
    PRE --> SAVE["ユーザー発言を履歴に保存"]
    SAVE --> R{"Intent Router<br/>router.py"}

    R -->|CHAT| CHAT["チャット<br/>(図4)"]
    R -->|VISION| VIS["画像・PDF・動画の理解<br/>Qwen3.8-27B + mmproj"]
    R -->|GENERATE| GEN["画像生成<br/>(図3)"]
    R -->|EDIT| EDT["画像編集<br/>参照画像 最大10枚 / マスク"]
    R -->|RESTORE| RES["元画像を新しい版としてコピー<br/>モデルは実行しない"]

    CHAT --> OUT
    VIS --> OUT
    GEN --> OUT
    EDT --> OUT
    RES --> OUT
    OUT["回答 / 画像を保存<br/>(履歴 + 画像の系譜)"] --> DONE["UI に描画<br/>SQLite → Drive に同期"]
```

## 3. 画像生成の流れ（固有キャラの外見検索つき）

「ブリーチの松本乱菊の画像を生成して」のように固有のキャラ・人物を指定すると、画像を作る前に外見を検索して「外見カード」を作り、プロンプトに反映します。参照画像が2枚以上あるときは、明示的に頼まれない限り検索しません。

```mermaid
flowchart TD
    A["生成 / 編集の依頼"] --> B{"固有名詞を含む?<br/>router.py<br/>(一般名詞は除外)"}
    B -- "いいえ" --> P
    B -- "はい" --> C{"Web検索が使える?"}
    C -- "いいえ / オフ" --> U["外見は未確認として扱う<br/>参照画像を優先"]
    U --> P
    C -- "はい" --> Q["検索クエリを作る<br/>作品名 + キャラ名 + official appearance<br/>+ character profile wiki<br/>(成人向けの場面指定はクエリから除く)"]
    Q --> S["検索して結果を絞る<br/>AIモデル配布サイトを除外<br/>Wikipedia / Fandom / ピクシブ百科事典を優先"]
    S --> CARD["外見カードを作る (Qwen3.8-27B)<br/>髪 / 目 / 服装 / 画風 など<br/>根拠のない特徴は UNKNOWN"]
    CARD --> P["画像プロンプトを英語に書き換える<br/>外見カードと矛盾する書き換えは却下<br/>(髪色の変更など)"]
    P --> SW["Chat モデル → 画像モデルへ切り替え<br/>(同時常駐できる GPU では入替なし)"]
    SW --> IMG["Qwen-Image-2.1 で生成<br/>既定 1280帯 / 30 steps<br/>ネガティブプロンプト + CFG 2.0"]
    IMG --> SV["画像を保存<br/>参照した出典を回答に表示"]
```

## 4. チャットの流れ（検索・調査エージェント）

メッセージに GitHub の URL があると調査エージェントが動き、なければ通常の Web 検索の判定に進みます。どちらも検索結果・読んだファイルは「データ」として扱い、その中の指示には従いません。

```mermaid
flowchart TD
    A["チャットの依頼"] --> B{"GitHub の URL を含む?<br/>QMC_AGENT=auto"}

    B -- "はい" --> L1["github_tree<br/>ファイル一覧"]
    L1 --> LOOP{"次の行動をモデルが JSON で選ぶ<br/>最大 14 回 / 重複は拒否"}
    LOOP -- "github_read" --> RD["ファイルを読む<br/>長いものはオフセットで続き"]
    LOOP -- "web_search / fetch" --> WS["検索 / ページ取得"]
    RD --> LOOP
    WS --> LOOP
    LOOP -- "finish / 上限" --> EV["調査ログ (読んだ内容) を<br/>システムプロンプトに追加"]

    B -- "いいえ" --> N{"Web検索が必要?<br/>QMC_WEB_SEARCH<br/>on / auto / off"}
    N -- "いいえ" --> ANS
    N -- "はい" --> QR["検索クエリを作る (最大3本)"]
    QR --> SR["検索 + 上位ページの本文取得"]
    SR --> CTX["検索結果を<br/>システムプロンプトに追加"]

    EV --> ANS["Qwen3.8-27B で回答<br/>思考モード: 任意"]
    CTX --> ANS
    ANS --> SRC["回答の末尾に出典 /<br/>読んだファイル一覧を表示"]
```

## 5. モデルと GPU の配置

起動時に GPU を判定してプロファイルを選びます。**メインの対象は A100 80GB** で、Chat と画像を同時に GPU に置く設計です。A100 40GB と L4 は、必要なときだけモデルを入れ替える方式のまま残しています。

```mermaid
flowchart TD
    GPU["GPU 判定<br/>gpu_manager.py"] --> P80["a100_80 (メイン)<br/>VRAM 70GiB 以上"]
    GPU --> P40["a100_40<br/>VRAM 38GiB 以上"]
    GPU --> PL4["l4 その他"]

    subgraph S80["a100_80: 同時常駐"]
        direction LR
        C80["Chat: Qwen3.8-27B Q8_K_L<br/>+ mmproj"]
        I80["Image: Qwen-Image-2.1<br/>bf16 / 全部 GPU"]
    end

    subgraph S40["a100_40: 入れ替え"]
        direction LR
        C40["Chat"] <--> I40["Image<br/>bf16 / CPU オフロード<br/>RAM に保持"]
    end

    subgraph SL4["l4: 入れ替え (Low VRAM)"]
        direction LR
        CL4["Chat<br/>ctx 16k"] <--> IL4["Image<br/>DiT int8 / CPU オフロード<br/>VAE タイリング"]
    end

    P80 --> S80
    P40 --> S40
    PL4 --> SL4
```

| | A100 80GB (`a100_80`) | A100 40GB (`a100_40`) | L4 (`l4`) |
| --- | --- | --- | --- |
| Chat と Image の同時常駐 | する | しない（入替） | しない（入替） |
| Chat の文脈長 | 32k | 32k | 16k |
| Image の精度 / 配置 | bf16 / 全部 GPU | bf16 / CPU オフロード | int8 / CPU オフロード |

> Chat を Q8_K_L に変更したあとの VRAM 使用量は **未計測** です。A100 80GB で `python -m qmc.bench` を再実行して確認します。Q8_K_L で動作が確認できているのは A100 80GB を想定した構成だけで、A100 40GB / L4 は未確認です。

`Model Manager` はモデルの中身を知らず、`load / unload / degrade` だけを扱います。メモリ不足（OOM）になったときは、配置を軽い方へ段階的に落として再実行します（全部 GPU → CPU オフロード → VAE タイリング → int8）。

## 6. データの保存

```mermaid
flowchart LR
    T["1ターンの終了"] --> W["SQLite に書き込み<br/>ローカルディスク"]
    W --> SNAP["スナップショットを<br/>Drive にミラー"]
    W --> IMGF["画像ファイル<br/>sessions/セッションID/..."]
    SNAP --> D[("Google Drive<br/>MyDrive/qwen-multimodal-colab/data")]
    IMGF --> D
    D -. "ランタイム再起動時に復元" .-> W
```

保存するのは、メッセージ・生成/編集した画像・画像の系譜（どの画像から作ったか）・使ったプロンプトと検索のメタ情報です。Drive が使えないときはランタイム内にだけ保存され、終了時に消えます。詳しくは [ADR-0002](adr/0002-sqlite-local-copy-with-drive-mirror.md)。

## 7. モジュール構成

```mermaid
flowchart TD
    subgraph 入口
        MAIN["__main__.py / app.py<br/>起動と組み立て"]
        COLAB["colab.py<br/>Colab 用セットアップ"]
        CFG["config.py<br/>設定と環境変数"]
    end

    UIm["ui.py<br/>Gradio (描画のみ)"]
    CT["controller.py<br/>1ターンの進行"]

    subgraph 判断と調査
        RT["router.py<br/>意図の判定"]
        PLC["policy.py<br/>コンテンツ方針"]
        SE["search_engine.py<br/>検索と外見検索"]
        AG["agent.py<br/>調査エージェント"]
    end

    subgraph エンジン
        CE["chat_engine.py"]
        VE["vision_engine.py"]
        IE["image_engine.py"]
        IMG["imaging.py<br/>PDF / 動画 / 画像の前処理"]
        ASRm["asr.py / tts.py"]
    end

    subgraph モデル管理
        MMm["model_manager.py"]
        GPUm["gpu_manager.py"]
        BE["backends/<br/>llama_server / qwen_image<br/>openai_compat / image_http / mock"]
    end

    subgraph 保存
        SES["session_manager.py"]
        HIS["history_manager.py"]
    end

    BENCH["bench.py<br/>VRAM / 速度の計測"]

    MAIN --> UIm
    MAIN --> CFG
    UIm --> CT
    CT --> RT
    CT --> PLC
    CT --> SE
    CT --> AG
    CT --> CE
    CT --> VE
    CT --> IE
    CT --> IMG
    CT --> ASRm
    CE --> MMm
    VE --> CE
    IE --> MMm
    MMm --> GPUm
    MMm --> BE
    CT --> SES
    SES --> HIS
    BENCH --> MMm
```

| 層 | ファイル | 役割 |
| --- | --- | --- |
| UI | `ui.py` | Gradio。ロジックは持たず、Controller のイベントを描画するだけ |
| Controller | `controller.py` | 1ターン = ルーティング → 実行 → 履歴保存。UI に依存しないイベントストリーム |
| Router | `router.py` | ルールベースの意図判定と、固有キャラの外見検索が必要かの判定 |
| 検索・調査 | `search_engine.py`, `agent.py` | Web 検索、外見検索（外見カード）、GitHub 等を読む調査エージェント |
| Engines | `chat_engine.py`, `vision_engine.py`, `image_engine.py` | 文脈の組み立て、生成パラメータ |
| Model Manager | `model_manager.py`, `gpu_manager.py` | 遅延ロード、入れ替え、OOM 回復、GPU プロファイル |
| Backends | `backends/*` | モデルの実体。ローカル / 外部 API / モックを差し替え可能 |
| 保存 | `session_manager.py`, `history_manager.py` | SQLite + Drive ミラー、画像の系譜 |

設計判断は [ADR](adr/) にあります。
