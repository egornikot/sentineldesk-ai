"""Static GitHub Pages build of SentinelDesk+ -> docs/plus/ (index.html + demo/results.json).

    py plus/build_static_plus.py

Same idea as scripts/build_static.py: records real runs of the combined pipeline for the demo cases and a
balanced sample of unseen test tickets, so the page works even when no live backend is connected.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PLUS = Path(__file__).resolve().parent
ROOT = PLUS.parent
sys.path[:0] = [str(ROOT), str(PLUS), str(ROOT / "scripts")]

from build_static import FLAG, key_of  # noqa: E402

from sentineldesk import data  # noqa: E402
from sentineldesk.config import MODELS_DIR, Settings  # noqa: E402
from sentineldesk_plus.pipeline import PlusPipeline  # noqa: E402

OUT = ROOT / "docs" / "plus"

if __name__ == "__main__":
    cfg = PLUS / "models" / "plus.json"
    pipe = PlusPipeline.build(Settings(), MODELS_DIR, cfg)
    quick = Settings()
    quick.llm_mode = "mock"
    scout = PlusPipeline.build(quick, MODELS_DIR, cfg)
    cases = [json.loads(l) for l in (ROOT / "evals" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    examples, results = [], {}
    for c in cases:
        pipe.radar.reset()
        r = pipe.run(c["subject"], c["body"])
        results[key_of(c["subject"], c["body"])] = r
        examples.append({"id": c["id"], "label": c["label"], "subject": c["subject"], "body": c["body"]})
        print(f"case {c['id']:<18} {r['decision']['route']}")
    test = data.load_split("test")
    pool = test[test["body"].str.len().between(80, 600)].sample(frac=1.0, random_state=7).head(300)
    cells: dict = {}
    for row in pool.itertuples():
        lang = row.language if row.language in FLAG else "en"
        scout.radar.reset()
        route = scout.run(row.subject, row.body)["decision"]["route"]
        if len(cells.setdefault((route, lang), [])) < 3:
            cells[(route, lang)].append(row)
    for (_, lang), rows in sorted(cells.items()):
        for row in rows:
            pipe.radar.reset()
            r = pipe.run(row.subject, row.body)
            r["ground_truth"] = {"ticket_id": row.ticket_id, "type": row.type, "queue": row.queue, "priority": row.priority}
            results[key_of(row.subject, row.body)] = r
            examples.append({"id": row.ticket_id, "subject": row.subject, "body": row.body,
                             "label": f"{FLAG[lang]} Unseen test {row.ticket_id}: {(row.subject or row.body)[:60]}"})
            print(f"test {row.ticket_id} {lang} {r['decision']['route']}")
    st = pipe.status()
    rep = PLUS / "reports"
    load = lambda f: json.loads((rep / f).read_text(encoding="utf-8")) if (rep / f).is_file() else {}  # noqa: E731
    compare = load("compare.json")
    bundle = {"built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "llm": f"{st['llm']} {st['model']}",
              "status": st, "metrics": load("metrics_plus.json"), "examples": examples, "results": results,
              "compare": {k: compare[k] for k in ("columns", "rows", "footnote")} if compare else None}
    (OUT / "demo").mkdir(parents=True, exist_ok=True)
    (OUT / "demo" / "results.json").write_text(json.dumps(bundle, ensure_ascii=False, default=str), encoding="utf-8")
    shutil.copy(PLUS / "static" / "dashboard.html", OUT / "index.html")
    print(f"wrote {OUT / 'index.html'} + demo/results.json: {len(examples)} examples")
