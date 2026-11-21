"""Send one ticket through the configured LLM and show the grounded draft (never prints keys).

    py scripts/smoke_llm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentineldesk.config import Settings  # noqa: E402
from sentineldesk.pipeline import Pipeline  # noqa: E402

SUBJECT = "Frage zur Datenexport-Funktion"
BODY = ("Hallo, wie kann ich die Berichte aus dem Dashboard als CSV exportieren? "
        "Die Exportschaltfläche finde ich nicht. Vielen Dank!")

if __name__ == "__main__":
    s = Settings()
    print(f"mode={s.llm_mode} gemini_key_set={bool(s.gemini_key)} model={s.gemini_model}")
    pipe = Pipeline.from_settings(s)
    t0 = time.time()
    r = pipe.run(SUBJECT, BODY)
    print(f"llm={pipe.llm.name} model={getattr(pipe.llm, 'model', '')} total={time.time() - t0:.1f}s")
    print("route:", r["decision"]["badge"], "|", "; ".join(r["decision"]["reasons"]))
    print("classification:", {h: r["classification"][h] for h in ("type", "queue", "priority")})
    print("top retrieved:", [(x["ticket_id"], x["similarity"], x["language"]) for x in r["retrieval"][:3]])
    d = r["draft"] or {}
    print("draft error:", d.get("error"))
    print("reply:\n" + (d.get("reply") or ""))
    print("claims:", json.dumps(d.get("claims"), ensure_ascii=False, indent=1))
    print("verification:", r["verification"])
    print("timings:", r["timings_ms"])
