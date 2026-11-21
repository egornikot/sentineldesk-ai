"""Make the public GitHub Pages dashboard triage ANY ticket live, served from this computer.

    py scripts/go_public.py            (or double-click go_public.bat)   original version
    py scripts/go_public.py --plus     (or go_public_plus.bat)           combined version

1. starts the SentinelDesk server on localhost:8000 (if it isn't running already)
2. opens a free Cloudflare quick tunnel to it (tools/cloudflared.exe, no account needed)
3. writes the tunnel address to docs/demo/backend.json and pushes it, so
   https://egornikot.github.io/sentineldesk-ai/ connects to it within ~1 minute
Keep this window open. Ctrl+C stops the tunnel; the page then falls back to the recorded examples.
Visitors are rate-limited (RATE_LIMIT_PER_HOUR / GLOBAL_LIMIT_PER_HOUR in .env).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLOUDFLARED = next((p for p in (ROOT / "tools" / "cloudflared.exe", ROOT.parent / "tools" / "cloudflared.exe") if p.is_file()),
                   shutil.which("cloudflared"))
GIT = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
PLUS_MODE = "--plus" in sys.argv  # the combined version: plus/run_plus.py -> .../sentineldesk-ai/plus/
BACKEND = ROOT / "docs" / ("plus/demo" if PLUS_MODE else "demo") / "backend.json"
PORT = 8001 if PLUS_MODE else int(os.environ.get("PORT", "8000"))
SERVER = ROOT / ("plus/run_plus.py" if PLUS_MODE else "run.py")
PAGE = "https://egornikot.github.io/sentineldesk-ai/" + ("plus/" if PLUS_MODE else "")


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def publish(api: str) -> None:
    BACKEND.write_text(json.dumps({"api": api, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    rel = BACKEND.relative_to(ROOT).as_posix()
    msg = f"Live backend: {api}" if api else "Live backend offline"
    subprocess.run([GIT, "add", rel], cwd=ROOT, check=False)
    subprocess.run([GIT, "commit", "-q", "-m", msg, "--", rel], cwd=ROOT, check=False)
    ok = subprocess.run([GIT, "push", "-q"], cwd=ROOT).returncode == 0
    print(("pushed " if ok else "PUSH FAILED (pull, then push docs/demo/backend.json by hand): ") + msg, flush=True)


def main() -> None:
    if not CLOUDFLARED:
        sys.exit("cloudflared not found: put cloudflared.exe in tools/ (github.com/cloudflare/cloudflared/releases)")
    procs = []
    if not server_up():
        print("starting SentinelDesk server ...", flush=True)
        procs.append(subprocess.Popen([sys.executable, str(SERVER), "--no-browser", "--port", str(PORT)], cwd=ROOT))
        for _ in range(120):
            if server_up():
                break
            time.sleep(1)
        else:
            sys.exit("server did not start")
    print("opening Cloudflare tunnel ...", flush=True)
    tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    procs.append(tunnel)
    url = None
    for line in tunnel.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
    if not url:
        sys.exit("tunnel failed to start")
    for _ in range(30):  # wait until the tunnel actually serves
        try:
            with urllib.request.urlopen(url + "/api/status", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(2)
    print(f"\n  Tunnel      : {url}", flush=True)
    publish(url)
    print(f"  Public page : {PAGE}  (live in ~1 min)")
    print("  Keep this window open. Ctrl+C to stop.\n", flush=True)
    try:
        for _ in tunnel.stdout:  # keep draining the tunnel log
            pass
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        publish("")


if __name__ == "__main__":
    main()
