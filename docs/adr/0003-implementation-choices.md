# ADR-0003: 仕様からの変更点（パッケージ構成・ルーター・モデル切替）

- Status: Accepted
- Date: 2026-09-28

仕様は「より良い方法があれば変更可、ただし理由を残す」としていたため、変更点をまとめる。

## 1. `src/*.py` → `src/qmc/` パッケージ
- 変更: 想定の `src/config.py` 等を `src/qmc/config.py` 等のパッケージにした（`python -m qmc` で起動）
- 理由: `src/` 直下のモジュール名（`config`, `ui` 等）は他ライブラリと衝突しやすい。相対importとエントリーポイントが使える
- デメリット: import が `from qmc.xxx import ...` と1段深くなる

## 2. 追加モジュール
| モジュール | 役割 |
| --- | --- |
| `controller.py` | 仕様の Chat Controller。UI非依存のイベントストリーム（将来 FastAPI からも使える） |
| `app.py` | 依存関係の組み立て（composition root） |
| `backends/` | `llama_server.py`（ローカル/リモートOpenAI互換）, `qwen_image.py`（diffusers）, `mock.py`（CPU用） |
| `imaging.py` | 画像の検証・縮小・data URI 化 |
| `colab.py` | Drive/Secrets/llama.cppビルドキャッシュ/起動 |
| `bench.py` | VRAM計測 |

## 3. Intent Router はルールベース
- 理由: L4 では LLM で判定すると「判定のためにChatモデルをロード → 画像モデルに切替」という余計なモデル入替が発生する。ルールなら即時・決定的・テスト可能
- 対策: 判定結果と理由をUIに表示し、手動モード（Chat/Vision/Generate/Edit）で上書きできる
- デメリット: 言い回しによっては誤判定する（テストケースで主要パターンを固定）

## 4. 画像プロンプトの LLM 最適化は「Chatモデルがロード済みのときだけ」（auto）
- 理由: L4 でプロンプト最適化のためだけに 27B を再ロードするのは遅い。UIで「常に」「使わない」も選べる

## 5. 編集時の seed
- 編集では親画像の系譜で使われた seed を必ず避ける（diffusers#14824: 同seed・同解像度だと指示が無視されたほぼコピーになる）

## 6. Stop の実装
- Gradio のジョブキャンセルではなく、`threading.Event` による協調キャンセル（LLMはストリームを閉じる、diffusersは `callback_on_step_end` で `_interrupt`）
- 理由: 生成途中のスレッドを強制終了すると VRAM やモデル状態が不定になるため
