#!/usr/bin/env python3
"""Start SentinelDesk: triage console + API.

    py run.py              uses the LLM keys in .env (Gemini, Claude as fallback), else the offline mock
    py run.py --mock       force the offline mock (no key, no cost)
    py run.py --port 8001  another port
    py run.py --no-browser
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit("SentinelDesk needs Python 3.10 or newer.")

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "plus")]

for stream in (sys.stdout, sys.stderr):  # Windows consoles can't print every character
    try:
        stream.reconfigure(errors="replace")
    except Exception:
        pass

from sentineldesk.config import MODELS_DIR, Settings  # noqa: E402
from sentineldesk_plus.pipeline import PlusPipeline  # noqa: E402
from sentineldesk_plus.server import serve  # noqa: E402

CFG = ROOT / "plus" / "models" / "plus.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="SentinelDesk server")
    parser.add_argument("--port", type=int)
    parser.add_argument("--host")
    parser.add_argument("--mock", action="store_true", help="use the offline mock instead of a real LLM")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    if args.mock:
        settings.llm_mode = "mock"
    settings.port = args.port or settings.port
    settings.host = args.host or settings.host
    if not CFG.is_file():
        sys.exit("Train first: py scripts/train.py, then py plus/train_plus.py")

    pipeline = PlusPipeline.build(settings, MODELS_DIR, CFG)
    try:
        httpd = serve(pipeline, settings.host, settings.port)
    except OSError:
        sys.exit(f"Port {settings.port} is busy. Close the other window, or run with --port 8001.")
    url = f"http://localhost:{settings.port}"
    st = pipeline.status()
    print("\n  SentinelDesk is running")
    print(f"  Console    : {url}")
    print(f"  LLM        : {st['llm']} {st['model']}" + ("  (add GEMINI_API_KEY or ANTHROPIC_API_KEY to .env)" if st["llm"] == "mock" else ""))
    print(f"  Classifier : {st['classifier']}")
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
