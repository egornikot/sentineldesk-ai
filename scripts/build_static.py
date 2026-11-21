"""Record a static demo of the base pipeline (classifiers only, no twin vote) into reports/baseline_static/.
The public page is built by plus/build_static_plus.py; this module also provides helpers it imports.

    py scripts/build_static.py [--per-cell 3] [--mock]

GitHub Pages can't run Python, so this records REAL runs of the full pipeline (trained models + the configured LLM)
for the demo cases plus a sample of unseen TEST tickets (balanced over route x language), and the dashboard replays
them. Test tickets carry their true labels so visitors can see where the model is right and wrong.
Then: GitHub repo Settings -> Pages -> Deploy from a branch -> main -> /docs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentineldesk import data  # noqa: E402
from sentineldesk.config import REPORTS_DIR, Settings  # noqa: E402
from sentineldesk.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "reports" / "baseline_static"
FLAG = {"en": "🇬🇧", "de": "🇩🇪"}
ROUTE_ICON = {"page_now": "🚨", "human_review": "👀", "draft_ready": "✅"}


def key_of(subject: str, body: str) -> str:
    return " ".join(f"{(subject or '').strip()}\n{(body or '').strip()}".split())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=3, help="test tickets per (route, language)")
    ap.add_argument("--scan", type=int, default=300)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    live = Settings()
    if a.mock:
        live.llm_mode = "mock"
    pipe = Pipeline.from_settings(live)
    quick = Settings()
    quick.llm_mode = "mock"
    scout = Pipeline.from_settings(quick)  # cheap pass to pick a balanced sample

    cases = [json.loads(l) for l in (ROOT / "evals" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    examples, results = [], {}
    for c in cases:
        r = pipe.run(c["subject"], c["body"])
        results[key_of(c["subject"], c["body"])] = r
        examples.append({"id": c["id"], "label": c["label"], "subject": c["subject"], "body": c["body"]})
        print(f"case {c['id']:<18} {r['decision']['route']}")

    test = data.load_split("test")
    pool = test[test["body"].str.len().between(80, 600)].sample(frac=1.0, random_state=7).head(a.scan)
    cells: dict = {}
    for row in pool.itertuples():
        lang = row.language if row.language in FLAG else "en"
        route = scout.run(row.subject, row.body)["decision"]["route"]
        if len(cells.setdefault((route, lang), [])) < a.per_cell:
            cells[(route, lang)].append(row)
    for (route, lang), rows in sorted(cells.items()):
        for row in rows:
            r = pipe.run(row.subject, row.body)
            r["ground_truth"] = {"ticket_id": row.ticket_id, "type": row.type, "queue": row.queue, "priority": row.priority}
            results[key_of(row.subject, row.body)] = r
            title = (row.subject or row.body)[:60]
            examples.append({"id": row.ticket_id, "subject": row.subject, "body": row.body,
                             "label": f"{FLAG[lang]} Unseen test {row.ticket_id}: {title}"})
            print(f"test {row.ticket_id} {lang} scout={route} live={r['decision']['route']}")

    status = pipe.status()
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    bundle = {"built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
              "llm": f"{status['llm']} {status['model']}", "status": status,
              "metrics": metrics, "examples": examples, "results": results}
    (OUT / "demo").mkdir(parents=True, exist_ok=True)
    (OUT / "demo" / "results.json").write_text(json.dumps(bundle, ensure_ascii=False, default=str), encoding="utf-8")
    shutil.copy(ROOT / "sentineldesk" / "static" / "dashboard.html", OUT / "index.html")
    (OUT / ".nojekyll").write_text("")
    print(f"\nwrote {OUT / 'index.html'} + demo/results.json: {len(examples)} examples")
