"""SentinelDesk+ server: the original server plus burst-radar and version-comparison routes."""
from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sentineldesk.server import make_handler

PLUS = Path(__file__).resolve().parent.parent
STATIC = PLUS / "static"
REPORTS = PLUS / "reports"


def make_plus_handler(pipeline: Any):
    Base = make_handler(pipeline)

    class Handler(Base):
        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "dashboard.html").read_bytes(), "text/html")
            if path == "/api/metrics":
                f = REPORTS / "metrics_plus.json"
                return self._send(200, json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {})
            if path == "/api/compare":
                f = REPORTS / "compare.json"
                return self._send(200, json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {})
            if path == "/api/burst":
                return self._send(200, pipeline.radar.snapshot())
            return super().do_GET()

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            if path == "/api/burst/simulate":
                pipeline.simulate_burst()
                return self._send(200, pipeline.radar.snapshot())
            if path == "/api/burst/reset":
                pipeline.radar.reset()
                return self._send(200, pipeline.radar.snapshot())
            return super().do_POST()

    return Handler


def serve(pipeline: Any, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_plus_handler(pipeline))
    httpd.daemon_threads = True
    return httpd
