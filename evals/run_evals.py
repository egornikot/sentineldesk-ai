"""Score the pipeline on the hand-written demo/safety cases in evals/cases.jsonl.   OWNER: D

    py evals/run_evals.py          (uses whatever models/index exist; mock LLM unless --live)
    py evals/run_evals.py --live   (use the real LLM if a key is configured)

Each case states what must happen (expect_route,
expect_route_not, expect_pii, expect_language); the script prints PASS/FAIL per case and a summary.
Before the classifier and index exist, route checks that need them may fail: that's expected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "plus"))

from sentineldesk.config import MODELS_DIR, Settings  # noqa: E402
from sentineldesk_plus.pipeline import PlusPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    if not args.live:
        settings.llm_mode = "mock"
    pipe = PlusPipeline.build(settings, MODELS_DIR, ROOT / "plus" / "models" / "plus.json")
    cases = [json.loads(l) for l in (ROOT / "evals" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    passed = 0
    for c in cases:
        pipe.radar.reset()  # cases are independent: no burst carried over between them
        r = pipe.run(c.get("subject"), c["body"])
        route, fails = r["decision"]["route"], []
        if "expect_route" in c and route != c["expect_route"]:
            fails.append(f"route {route} != {c['expect_route']}")
        if "expect_route_not" in c and route == c["expect_route_not"]:
            fails.append(f"route should not be {route}")
        missing = set(c.get("expect_pii", [])) - set(r["pii"]["found"])
        if missing:
            fails.append(f"PII not masked: {sorted(missing)}")
        if "expect_language" in c and r["ticket"]["language"] != c["expect_language"]:
            fails.append(f"language {r['ticket']['language']} != {c['expect_language']}")
        passed += not fails
        print(f"{'PASS' if not fails else 'FAIL'}  {c['id']:<18} route={route:<13} {r['timings_ms']['total']:>7} ms  {'; '.join(fails)}")
    print(f"\n{passed}/{len(cases)} passed  ·  engine: {pipe.status()}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
