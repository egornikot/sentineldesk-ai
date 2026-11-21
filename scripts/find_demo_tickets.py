"""Find unseen TEST tickets that make good live-demo examples (by route and language).

    py scripts/find_demo_tickets.py --lang de --route draft_ready --n 5

Uses only subject + body (what a judge would paste). Mock LLM, so no API cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentineldesk import data  # noqa: E402
from sentineldesk.config import Settings  # noqa: E402
from sentineldesk.pipeline import Pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="de")
    ap.add_argument("--route", default="draft_ready")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--scan", type=int, default=400)
    a = ap.parse_args()
    s = Settings()
    s.llm_mode = "mock"
    pipe = Pipeline.from_settings(s)
    test = data.load_split("test")
    pool = test[(test["language"] == a.lang) & (test["body"].str.len().between(80, 420))].head(a.scan)
    found = 0
    for row in pool.itertuples():
        r = pipe.run(row.subject, row.body)
        if r["decision"]["route"] == a.route:
            top = r["retrieval"][0]
            print(f"--- {row.ticket_id}  top {top['ticket_id']} sim {top['similarity']}  "
                  f"pred {r['classification']['queue']['label']}")
            print("SUBJECT:", row.subject)
            print("BODY:", row.body.replace("\n", " ")[:400])
            found += 1
            if found >= a.n:
                break
    print(f"\nfound {found} in {len(pool)} scanned")
