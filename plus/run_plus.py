#!/usr/bin/env python3
"""Start SentinelDesk+ (the combined version) on port 8001. The original stays on run.py / port 8000.

    py plus/run_plus.py [--port 8001] [--mock] [--no-browser]
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

PLUS = Path(__file__).resolve().parent
ROOT = PLUS.parent
sys.path[:0] = [str(ROOT), str(PLUS)]

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass

from sentineldesk.config import MODELS_DIR, Settings  # noqa: E402
from sentineldesk_plus.pipeline import PlusPipeline  # noqa: E402
from sentineldesk_plus.server import serve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SentinelDesk+ server")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    settings = Settings()
    if a.mock:
        settings.llm_mode = "mock"
    settings.host = a.host or settings.host
    cfg = PLUS / "models" / "plus.json"
    if not cfg.is_file():
        sys.exit("run plus/train_plus.py first")
    pipe = PlusPipeline.build(settings, MODELS_DIR, cfg)
    try:
        httpd = serve(pipe, settings.host, a.port)
    except OSError:
        sys.exit(f"Port {a.port} is busy: use --port")
    url = f"http://localhost:{a.port}"
    st = pipe.status()
    print(f"\n  SentinelDesk+ (combined) is running\n  Console : {url}\n  LLM     : {st['llm']} {st['model']}\n"
          f"  Twins   : k={st['twin_vote']['k']}\n  Press Ctrl+C to stop\n")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
