# ADR-0002: SQLite はローカルに置き、Google Drive へスナップショットをミラーする

- Status: Accepted
- Date: 2026-09-28

## Context
仕様では「SQLite推奨」「Google Driveをマウントすれば再起動後に復元」とされていた。
しかし Colab の Drive は FUSE マウントで、SQLite が前提とするファイルロック・fsync の保証が弱く、
ライブDBを直接置くと破損や "database is locked" のリスクがある。

## Decision
- ライブDBはローカルディスク（既定 `/tmp/qmc/history.db`）に置く
- 各ターン終了時に `sqlite3` の backup API で一時ファイルに書き出し、`os.replace` で Drive 上の `history.db` に**アトミックに**置き換える
- 起動時、ローカルDBが無い／Drive側が新しい場合は Drive から復元する
- 起動時に `PRAGMA integrity_check` を行い、破損していれば退避（`*.corrupt-<ts>.db`）→ Driveのバックアップから復元 → 無ければ新規作成。UIに通知する
- 画像ファイル（PNG）は Drive 上の `data/sessions/<session_id>/images/` に直接保存（tmp → rename）
- journal_mode は WAL ではなく DELETE（単一ファイルなのでコピー復元が常に安全）

## Consequences
- 👍 Drive の不安定さでDBが壊れない。Colab切断時も最後のターンまでは Drive に残る
- 👍 仕様どおり SQLite（sessions / messages / images / generations / model_settings）で履歴を検索・集計できる
- 👎 ターン途中で切断された場合、そのターンの途中経過は失われる
- 👎 複数ブラウザから同時に書き込む用途（マルチユーザー）には向かない → 将来は PostgreSQL 等へ
