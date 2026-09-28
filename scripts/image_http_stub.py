"""CPU-only HTTP image worker stub. Run: python scripts/image_http_stub.py --port 8013"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path not in {"/v1/images/generations", "/v1/images/edits"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 80 * 1024**2:
            self.send_error(413)
            return
        body = self.rfile.read(length)
        if self.path.endswith("generations"):
            try:
                data = json.loads(body)
                label = str(data.get("prompt", ""))[:80]
            except ValueError:
                self.send_error(400)
                return
        else:
            if "multipart/form-data" not in self.headers.get("Content-Type", ""):
                self.send_error(400)
                return
            label = "MOCK EDIT"
        image = Image.new("RGB", (512, 512), (52, 47, 64))
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), label, fill="white")
        stream = io.BytesIO()
        image.save(stream, "PNG")
        payload = json.dumps({"data": [{"b64_json": base64.b64encode(stream.getvalue()).decode()}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def make_server(host: str = "127.0.0.1", port: int = 8013):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    args = parser.parse_args()
    print(f"Mock image worker on http://{args.host}:{args.port}", flush=True)
    make_server(args.host, args.port).serve_forever()
