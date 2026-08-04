"""Local dashboard for the sidecar run store; deliberately no trading dependencies."""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from parabolic.platform import RunStore


PAGE = b'''<!doctype html><title>Parabolic Runs</title><style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:.5rem;text-align:left}pre{white-space:pre-wrap}</style><h1>Parabolic strategy runs</h1><table><thead><tr><th>Run</th><th>Status</th><th>Command</th><th>Strategy result</th><th>Agent metadata</th></tr></thead><tbody id="runs"></tbody></table><script>fetch('/api/runs').then(r=>r.json()).then(rows=>document.querySelector('#runs').innerHTML=rows.map(r=>`<tr><td>${r.run_id}</td><td>${r.status}</td><td>${r.argv.join(' ')}</td><td><pre>${JSON.stringify(r.result||{},null,1)}</pre></td><td><pre>${JSON.stringify(r.metadata,null,1)}</pre></td></tr>`).join(''))</script>'''


def make_handler(state_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: object, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _store(self) -> RunStore:
            return RunStore(state_dir)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
                return
            store = self._store()
            try:
                if self.path == "/api/runs":
                    self._send(200, [asdict(run) for run in store.list(200)])
                    return
                if self.path.startswith("/api/runs/"):
                    run = store.get(self.path.rsplit("/", 1)[-1])
                    self._send(200 if run else 404, asdict(run) if run else {"error": "not found"})
                    return
                self._send(404, {"error": "not found"})
            finally:
                store.close()

        def do_PATCH(self) -> None:  # noqa: N802
            if not self.path.startswith("/api/runs/"):
                self._send(404, {"error": "not found"})
                return
            try:
                metadata = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if not isinstance(metadata, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "metadata must be a JSON object"})
                return
            store = self._store()
            try:
                run = store.update_metadata(self.path.rsplit("/", 1)[-1], metadata)
                self._send(200 if run else 404, asdict(run) if run else {"error": "not found"})
            finally:
                store.close()

        def do_DELETE(self) -> None:  # noqa: N802
            store = self._store()
            try:
                deleted = store.delete(self.path.rsplit("/", 1)[-1]) if self.path.startswith("/api/runs/") else False
                self._send(204 if deleted else 404, b"")
            finally:
                store.close()

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve(state_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(state_dir))
    print(f"Parabolic dashboard: http://{host}:{port}")
    server.serve_forever()
