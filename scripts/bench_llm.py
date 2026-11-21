"""Time each Gemini model on a realistic drafting request (no classifier/embeddings needed).

    py scripts/bench_llm.py [model ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from sentineldesk import verify  # noqa: E402
from sentineldesk.config import MODELS_DIR, Settings  # noqa: E402
from sentineldesk.generate import draft_reply  # noqa: E402
from sentineldesk.llm import GeminiLLM  # noqa: E402

if __name__ == "__main__":
    s = Settings()
    meta = pd.read_parquet(MODELS_DIR / "index_meta.parquet")
    vpn = meta[meta["body"].str.contains("VPN", case=False)].head(4)
    retrieval = [{**r, "similarity": 0.9} for r in vpn.to_dict("records")]
    ticket = {"subject": "VPN keeps disconnecting", "body": "Since this morning our VPN client disconnects every few minutes for some users. How can we fix it?", "language": "en"}
    for model in sys.argv[1:] or ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]:
        llm = GeminiLLM(s.gemini_key, model, s.gemini_thinking, fallbacks=[])
        t0 = time.time()
        d = draft_reply(llm, ticket, retrieval, {"route": "draft_ready"})
        v = verify.verify(d, retrieval)
        print(f"{model:<24} {time.time() - t0:5.1f}s  error={d['error'] and d['error'][:80]}  claims={len(d['claims'])} "
              f"grounded={v['grounding_rate']}  problems={v['problems'][:2]}")
