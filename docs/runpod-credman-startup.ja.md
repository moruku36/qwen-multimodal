# 通常start / serveのWindows provider連携

既存の起動入口 `scripts/runpod_trial.py` に、Windows v2 providerを読む経路を
追加した。従来の明示入力経路は維持する。選択したv2経路が拒否された場合は
入力経路や別資格へ自動切替しない。defaultは無効。

起動時に `--owner-start`、既存の非秘密 `--config`、明示的な
`--operations-source` と、次の4引数を指定する。

- `--credential-approval`: 本人が別途作った起動専用の承認JSON。
- `--credential-factory-source`: レビューしたFactory作業場所。
- `--credential-launcher-sha256`: Factoryの `runpod_startup_credential.py` のSHA256。
- `--credential-claims`: Factoryの親にある既存 `runpod-startup-claims`。

実行は本人Windowsターミナルのisolated Python（`-I -B`）から行う。
実際の値や承認JSONは共有・コミットしない。詳細な承認schemaはFactoryの
`docs/security/runpod-startup-credential.ja.md` を参照する。

アカウント確認用の承認は受け付けない。`start` / `serve` に一致する単回承認で
設定・起動ソース・operations全Pythonソース・Python・本人SID・絶対パスを拘束する。
今回のコード反映承認は、実読出し・API送信・新Pod作成を許可するものではない。

providerだけを既存v2ストアから1回読む。他の既存資格は子の本人コンソールで
非表示入力する。接続先の本人確認と個別削除承認も同じ子で扱う。秘密値は
親へ返さず、プロセス引数・環境・ログ・公開結果に保存しない。

受付は最大300秒。資格読出しを親が27秒で監督し、試験・後処理は既存の
5400秒＋600秒計画で最大6000秒、終了回収猶予3秒。受付期間が失効したら
試験開始前に拒否する。試験開始後は承認されたsession期限を使う。

強制終了だけではPodの停止・削除・0ドル/時を確認できない。試験開始後の
期限超過・失敗・終了未確認ではcleanupが必要な状態を維持する。OSの停止や
子孫プロセスの終了は保証しない。自動再試行、永続許可、新規資格登録はない。

模擬SID/ストアで、実workerの公開ソース捕捉から通常provider、実試験制御、
模擬REST/SSH、本人接続確認・個別削除承認、readbackと不在確認までを検証する。
実コンソール・実キー・外部送信・GPUの通常起動は未検証。
