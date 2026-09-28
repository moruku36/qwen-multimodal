# リモート Image HTTP API

`QMC_IMAGE_BASE_URL=http://worker-host:8013` を設定すると、Chat UI はローカルの Diffusers を読み込まず、画像処理を HTTP ワーカーへ送ります。`QMC_IMAGE_API_KEY` があれば `Authorization: Bearer <key>` を付けます。通信には TLS とアクセス制御を持つワーカーを使ってください。

## 生成

`POST /v1/images/generations`、`Content-Type: application/json`。

```json
{"prompt":"a cat", "width":1024, "height":1024, "steps":40, "seed":123, "negative_prompt":""}
```

## 編集

`POST /v1/images/edits`、`multipart/form-data`。テキスト項目は `prompt`, `width`, `height`, `steps`, `seed`, `negative_prompt`。添付項目 `images` を参照順に繰り返します。マスク使用時は白が編集対象の PNG を `mask` として 1 枚添付します。最後の参照画像が編集元です。実際のマスク制御はワーカー側で実装します。

## 応答と失敗

両方とも HTTP 200 で `{"data":[{"b64_json":"<PNG/JPEGのbase64>"}]}`、または `{"data":[{"url":"https://..."}]}` を返します。失敗時は適切な 4xx/5xx を返してください。UI は応答待ちを最大 600 秒とし、エラーを画面へ表示します。Stop は次の処理を止めますが、進行中の HTTP 要求はタイムアウトまで続く場合があります。

契約確認用の CPU モックは `python scripts/image_http_stub.py --port 8013` で起動できます。画像の内容はプロンプトを反映した本物の生成結果ではありません。RunPod などの本番ワーカーはこの API を実装してください。
