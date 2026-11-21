# SentinelDesk: build plan

The build has 6 hours. Get the core loop working end to end first, then improve it. Bonus features come only after the dashboard demo works.

## 1. Repo layout

```
sentineldesk/
  run.py                    start the server and dashboard
  start.bat, check.bat      one-click start and key check
  .env.example              keys + thresholds
  sentineldesk/
    config.py               env and settings loading
    llm.py                  Claude client + offline mock fallback
    guard.py                PII masking before the LLM
    data.py                 load CSV, clean, dedupe, grouped 80/20 split
    classify.py             type/queue/priority models + critical-incident detector
    retrieve.py             multilingual embedding index over the train split only
    gate.py                 PAGE NOW / HUMAN REVIEW / DRAFT READY + reasons
    generate.py             LLM draft as structured JSON claims with citations
    verify.py               citation + excerpt checks, promise/amount/SLA flags
    pipeline.py             runs every stage, records per-stage latency
    db.py                   SQLite store for processed tickets + review queue
    server.py               stdlib HTTP API + dashboard
    static/dashboard.html   triage console page
  scripts/
    prepare_data.py         → data/processed/{train,dev,test}.parquet
    train.py                → models/*.joblib, models/index.npy
    evaluate.py             → reports/metrics.json, reports/metrics.md, reports/*.png
  evals/
    cases.jsonl             hand-written demo + safety cases
    run_evals.py            scorer for those cases
  tests/
  reports/                  committed metrics and charts
  data/, models/            not committed (gitignored)
```


## 1b. How we work together

Getting started:

1. `git clone https://github.com/egornikot/sentineldesk-ai.git`, then `cd sentineldesk-ai`.
2. Run `py run.py --mock` to check that the console opens, and `py -m unittest discover -s tests` to check that the tests pass.
3. Each module's docstring names its **OWNER** (A/B/C/D) and its exact contract. Work in your own files.

Branches:

- Create one with `git checkout -b a-data`, `b-retrieval`, `c-dashboard` or `d-evals`.
- Commit small and often, push, and open a PR into `main`. Merge when the tests pass.
- If you need to change the contract in section 2, say so in the team chat first, because three modules depend on it.

Rules for the repo:

- Never commit `.env`, `data/` or `models/`. The `.gitignore` already blocks them.
- The large files are rebuilt by `scripts/`, so they don't belong in git.

## 2. Interface contract

Every module reads and writes this shape. Agree on it first so the stages can be built in parallel.

```json
{
  "ticket":  {"subject": "Hospital VPN not working", "body": "...", "language": "en"},
  "pii":     {"masked": true, "found": ["email"]},
  "classification": {
    "type":     {"label": "Incident",   "confidence": 0.96},
    "queue":    {"label": "IT Support", "confidence": 0.71},
    "priority": {"label": "high",       "confidence": 0.88},
    "critical_score": 0.93
  },
  "retrieval": [
    {"ticket_id": "TKT-04812", "similarity": 0.82, "subject": "...", "body": "...", "answer": "...",
     "type": "Incident", "queue": "IT Support", "priority": "high", "language": "en"}
  ],
  "decision": {
    "route": "page_now | human_review | draft_ready",
    "reasons": ["critical_score 0.93 >= 0.40", "impact: patient records"]
  },
  "draft": {
    "language": "en",
    "reply": "...",
    "claims": [{"sentence": "...", "ticket_id": "TKT-04812", "excerpt": "...", "verified": true}],
    "flags": ["refund amount not supported by evidence"]
  },
  "timings_ms": {"preprocess": 3, "classify": 12, "retrieve": 20, "gate": 1, "generate": 1800, "verify": 5, "total": 1841}
}
```

- The dataset has no ticket IDs, so `prepare_data.py` assigns stable `TKT-xxxxx` IDs, the format the official brief uses.
- On the dashboard, the routes show as the brief's badges: `page_now` = **Escalate to On-Call Engineer**, `human_review` = **Human Review**, `draft_ready` = **Auto-Draft Ready**. Every badge shows its reasons.
- On `page_now`, the customer-facing reply is a fixed acknowledgment. The retrieved cases are shown to the engineer as an internal brief and are not sent to the customer.

## 3. Decision rules

The thresholds are tuned on the dev split, never on test.

| Route | When |
|---|---|
| **PAGE NOW** | Predicted Incident + high priority, **or** `critical_score` ≥ the threshold tuned for ~90%+ recall, **or** an impact rule fires (patients/clinical, all users/site down, security breach, data loss), with negation handling |
| **HUMAN REVIEW** | Any confidence below its threshold, **or** top retrieval similarity below the evidence threshold, **or** the ticket asks for a refund/amount/guarantee/legal/SLA promise, **or** any drafted claim fails verification |
| **DRAFT READY** | None of the above |

The cost of the extra recall goes in the report: the number of other tickets each tier flags, with "predicted labels only" and "+ critical detector" side by side.

## 4. Six-hour schedule

| Time | Data + models (A) | Retrieval + LLM (B) | Gate + dashboard (C) | Pitch + evals (D) |
|---|---|---|---|---|
| **H1** | `data.py`, `prepare_data.py`: pick one CSV, fill blank subjects, dedupe, grouped split 72/8/20 (train/dev/test) | Build `llm.py`, `config.py`, `guard.py`; get the LLM + mock running | Build `server.py`, `db.py`, `dashboard.html`, `run.py`; stub `/api/triage` returning the contract above | Write `evals/cases.jsonl`: VPN outage, German request, $500 refund guarantee, prompt injection, negated "not urgent", PII |
| **H2** | TF-IDF + LR baseline for type/queue/priority, critical detector; save models | `retrieve.py`: multilingual embeddings over train; top-5 + similarity | `gate.py` with reasons; wire `pipeline.py` with timings | Pre-test the demo cases against the index; pick the German example |
| **H3** | Try embedding features / stacking; calibrate confidences | `generate.py`: JSON claims prompt, EN/DE | Dashboard: classification bars, route badge + reasons, similar-tickets table | README draft |
| **H4** | `evaluate.py`: macro-F1, recall vs. flag rate, calibration, latency | `verify.py`: ID check, fuzzy excerpt match, promise flags; one retry, else review | Evidence panel: click a sentence → its excerpt + ticket ID; review queue | Pitch outline, screenshots |
| **H5** | Final metrics → `reports/` | Grounding rate on 50–100 test tickets | Metrics tab; latency display; polish | Rehearse the 5-minute demo |
| **H6** | Freeze. Test on unseen tickets, fix bugs, finish the README, push, rehearse. | | | |

With 3 people, A also takes D's evals and B takes the README.

## 5. Bonuses (only after H5 is green)

1. **Cross-language retrieval**: free with multilingual embeddings. Demo: a German ticket retrieves an English precedent.
2. **PII masking**: `guard.py` masks emails, phone numbers, IP addresses, account numbers and IBANs, as the brief lists.
3. **Outage bursts**: cluster tickets that arrive close together by embedding similarity. The CSV has no timestamps, so this must be labeled a **simulated stream**.

## 6. Rules we don't break

- Use one CSV, `aa_dataset-tickets-multi-lang-5-2-50-version.csv`. The brief's `load_dataset` snippet merges every file in the repository, and those files share thousands of duplicate tickets, which would leak across the split. We go stricter than the brief: duplicates are grouped into the same split.

- Split before anything else. Never fit, tune or index on test.
- The LLM never gets the final say on safety: the gate and the verifier are plain code.
- No invented policies, amounts, SLAs or guarantees. If the evidence doesn't contain it, the ticket goes to review.
- Never commit `.env`, data or model files.
