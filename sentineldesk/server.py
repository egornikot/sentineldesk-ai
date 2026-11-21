"""HTTP server: triage console + JSON API (stdlib only).        OWNER: C (gate + dashboard)

ThreadingHTTPServer, one static dashboard page, JSON routes.

    GET  /                 dashboard (static/dashboard.html)
    GET  /api/status       which engines are loaded (trained classifier? index? Claude or mock?)
    GET  /api/metrics      reports/metrics.json written by scripts/evaluate.py (or {})
    GET  /api/examples     demo tickets from evals/cases.jsonl
    POST /api/triage       {"subject": str, "body": str, "llm"?: "gemini"|"claude"} -> pipeline result (docs/PLAN.md contract)
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import REPORTS_DIR, ROOT

STATIC = Path(__file__).resolve().parent / "static"
CASES = ROOT / "evals" / "cases.jsonl"


class RateLimiter:
    """Per-visitor sliding window, so a public deployment can't drain the LLM credits.

    RATE_LIMIT_PER_HOUR (default 0 = off locally); GLOBAL_LIMIT_PER_HOUR caps all visitors together.
    """

    def __init__(self, per_visitor: int, global_cap: int) -> None:
        self.per_visitor, self.global_cap = per_visitor, global_cap
        self.hits: dict = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, visitor: str) -> bool:
        if self.per_visitor <= 0:
            return True
        now = time.time()
        with self.lock:
            for key in (visitor, "*"):
                q = self.hits[key]
                while q and now - q[0] > 3600:
                    q.popleft()
            if len(self.hits[visitor]) >= self.per_visitor:
                return False
            if self.global_cap > 0 and len(self.hits["*"]) >= self.global_cap:
                return False
            self.hits[visitor].append(now)
            self.hits["*"].append(now)
            return True


def make_handler(pipeline: Any):
    limiter = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_HOUR", "0") or 0),
                          int(os.environ.get("GLOBAL_LIMIT_PER_HOUR", "0") or 0))

    class Handler(BaseHTTPRequestHandler):
        def _visitor(self) -> str:  # behind a proxy/tunnel the real client is in X-Forwarded-For
            fwd = self.headers.get("X-Forwarded-For", "")
            return fwd.split(",")[0].strip() if fwd else self.client_address[0]

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
            if "/api/triage" in (str(args[0]) if args else ""):
                super().log_message(fmt, *args)

        def do_HEAD(self) -> None:  # health checks
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

        def _cors(self) -> None:
            # the public GitHub Pages dashboard calls this server through a tunnel (scripts/go_public.py)
            origin = self.headers.get("Origin", "")
            if origin.endswith(".github.io") or origin.startswith(("http://localhost", "http://127.0.0.1")):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:  # CORS preflight
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(self, code: int, payload: Any, content_type: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + ("; charset=utf-8" if "json" in content_type or "html" in content_type else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "dashboard.html").read_bytes(), "text/html")
            if path == "/api/status":
                return self._send(200, pipeline.status())
            if path == "/api/metrics":
                f = REPORTS_DIR / "metrics.json"
                return self._send(200, json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {})
            if path == "/api/examples":
                cases = []
                if CASES.is_file():
                    for line in CASES.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            c = json.loads(line)
                            cases.append({k: c.get(k) for k in ("id", "label", "subject", "body")})
                return self._send(200, cases)
            return self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/api/triage":
                return self._send(404, {"error": "not found"})
            if not limiter.allow(self._visitor()):
                return self._send(429, {"error": "Demo limit reached: please try again in a few minutes."})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(min(length, 200_000)) or b"{}")
                text = str(body.get("body") or "").strip()
                if not text:
                    return self._send(400, {"error": "body is required"})
                choice = body.get("llm") if body.get("llm") in ("gemini", "claude") else None
                return self._send(200, pipeline.run(str(body.get("subject") or "")[:500], text[:10000], choice))
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def serve(pipeline: Any, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(pipeline))
    httpd.daemon_threads = True
    return httpd
