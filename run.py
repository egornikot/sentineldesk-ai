#!/usr/bin/env python3
"""Start SentinelDesk: triage console + API.

    py run.py              use Claude if ANTHROPIC_API_KEY is set in .env, else the offline mock
    py run.py --mock       force the offline mock (no key, no cost)
    py run.py --port 8001  another port
    py run.py --no-browser
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

if sys.version_info < (3, 10):
    sys.exit("SentinelDesk needs Python 3.10 or newer.")

for stream in (sys.stdout, sys.stderr):  # Windows consoles can't print every character
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass

from sentineldesk.config import Settings
from sentineldesk.pipeline import Pipeline
from sentineldesk.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelDesk server")
    parser.add_argument("--port", type=int)
    parser.add_argument("--host")
    parser.add_argument("--mock", action="store_true", help="use the offline mock instead of Claude")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    if args.mock:
        settings.llm_mode = "mock"
    settings.port = args.port or settings.port
    settings.host = args.host or settings.host

    pipeline = Pipeline.from_settings(settings)
    try:
        httpd = serve(pipeline, settings.host, settings.port)
    except OSError:
        sys.exit(f"Port {settings.port} is busy. Close the other window, or run with --port 8001.")
    url = f"http://localhost:{settings.port}"
    st = pipeline.status()
    print("\n  SentinelDesk is running")
    print(f"  Console    : {url}")
    print(f"  LLM        : {st['llm']} {st['model']}" + ("  (add ANTHROPIC_API_KEY to .env for Claude)" if st["llm"] == "mock" else ""))
    print(f"  Classifier : {st['classifier']}" + ("  (run scripts/train.py)" if st["classifier"] == "fallback" else ""))
    print(f"  Retrieval  : {st['retriever']}")
    print("  Press Ctrl+C to stop\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
