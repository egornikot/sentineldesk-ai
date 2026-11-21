# SentinelDesk

An AI incident triage copilot for customer support. For every ticket it:

1. **Classifies** the ticket's type, queue and priority, each with a calibrated confidence score.
2. **Decides whether a human is needed**: page someone now, send to human review, or mark an AI draft as ready. Every decision lists its reasons.
3. **Retrieves** the 5 most similar resolved tickets from the historical data, with similarity scores. Matching works across languages.
4. **Drafts a reply** in English or German, matching the customer. Every instruction in the reply cites the past ticket it comes from, and the code checks each citation before the draft is shown.

> **Core idea:** urgency and evidence are two separate decisions. A hospital VPN outage is paged to an engineer immediately, even when there is no known fix. A routine billing ticket can go to the right queue with high confidence while the AI declines to promise a refund that no past case supports.


## Two versions

| | Run | Live demo |
|---|---|---|
| **SentinelDesk** (original, this page) | `py run.py` → localhost:8000 | https://egornikot.github.io/sentineldesk-ai/ |
| **SentinelDesk+** (combined: ours + teammate's Helios + akshita's branch) | `py plus/run_plus.py` → localhost:8001 | https://egornikot.github.io/sentineldesk-ai/plus/ |

SentinelDesk+ adds a twin vote over the most similar training tickets, temperature calibration, an outage burst
radar, a "what left your network" PII view, placeholder cleanup, and an honest paraphrase-grouped evaluation.
See [plus/README.md](plus/README.md) and the side-by-side results in
[plus/reports/COMPARE_RESULTS.md](plus/reports/COMPARE_RESULTS.md).

## Results on the held-out test set (5,718 tickets never seen in training)

| Metric | Result |
|---|---|
| Macro-F1: type / queue / priority | **0.859** / 0.571 / 0.587 |
| High-priority incident recall (full gate) | **89.9%** (886 of 986), with 26.6% of other tickets paged |
| â€¦using confident labels only | 51.4% recall: the separate critical detector is what closes the gap |
| Calibration error (ECE, type) | 0.032: a confidence of 0.8 really means about 80% correct |
| Routes | 37.5% page now Â· 53.4% human review Â· 9.0% draft ready |
| Grounding (live Gemini, demo cases) | 100% of cited claims verified against the retrieved tickets |
| Latency | classify + retrieve â‰ˆ 60 ms; end to end â‰ˆ 1.3 s with Gemini 3.5 Flash-Lite |
| Demo and safety cases | 8 of 8 pass (`evals/run_evals.py --live`) |

The full report is in [reports/metrics.md](reports/metrics.md), with the calibration plot in `reports/calibration.png`. Per-ticket predictions are in `reports/predictions/sentineldesk_ml_test.csv`.

**No leakage.** Duplicate texts are grouped, then split 72/8/20 (train/dev/test) and stratified by type and priority. Models and thresholds are chosen on dev only. Only train tickets are indexed for retrieval. The exact assignment is in `splits/split_assignment.csv`, so other models can be scored on the same test set (see [docs/COMPARE.md](docs/COMPARE.md)).

## Pipeline

```
ticket (subject + body)
  â†’ preprocess   mask PII (email, phone, IP, IBAN, cards, accounts, keys), detect EN/DE
  â†’ classify     type / queue / priority + confidence, critical-incident score    (~50 ms)
  â†’ retrieve     top-5 similar resolved TRAIN tickets, multilingual embeddings
  â†’ gate         PAGE NOW / HUMAN REVIEW / DRAFT READY, with reasons
  â†’ generate     LLM reply as JSON claims, each with a ticket ID + verbatim excerpt
  â†’ verify       each excerpt must fuzzy-match the cited ticket; flag refunds/amounts/SLAs/legal promises
  â†’ dashboard    the full trace, per-stage latency, evidence panel
```

**Gate rules.**
- **Page now:** a confident Incident + high label, a critical score over the tuned threshold, or severe impact words. Impact words are clinical, outage, security or data loss, and they must come with urgency or a second signal. Negations such as "not urgent" are ignored.
- **Human review:** low confidence on any head, weak precedent (best match below the 10th percentile of dev similarity), a promise request, or a draft that fails verification.
- **Draft ready:** everything else.

**Models.** Three candidates per head are compared on dev:
- TF-IDF word and character n-grams with a calibrated linear SVM
- multilingual MiniLM embeddings with logistic regression
- both combined

The winners are TF-IDF for type and priority, and combined for queue and the critical detector. All thresholds are tuned on dev: critical-incident recall â‰¥ 90%, and â‰¥ 75% accuracy above each confidence cut.

**LLM.** Gemini 3.5 Flash-Lite is the default, with a fallback to Gemini 3.8 Flash. If both fail or take longer than 12 s, Claude takes over (when `ANTHROPIC_API_KEY` is set). With no key, an offline mock drafts from the top precedent. Page-now tickets get an internal brief plus a short customer acknowledgement, not a promised fix.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
copy .env.example .env        # add GEMINI_API_KEY and/or ANTHROPIC_API_KEY (or leave empty: mock mode)
py scripts/prepare_data.py    # download, clean, dedupe, grouped split
py scripts/train.py           # classifiers + thresholds + retrieval index (~10 min first time)
py scripts/evaluate.py        # reports/metrics.json + .md, calibration plot, test predictions
py run.py                     # console at http://localhost:8000 (or double-click start.bat)
```

Checks:

```powershell
py -m unittest discover -s tests       # 14 offline tests
py evals/run_evals.py --live           # 8 demo + safety cases with the real LLM
py scripts/compare.py                  # compare/combine with another model's predictions
```

## Tuning knobs

- `TARGET_RECALL` in `scripts/train.py` trades critical-incident recall against how many other tickets get paged.
- `TARGET_ACCURACY` and `WEAK_EVIDENCE_PCT` control how many tickets reach draft ready (9% today; the conservative default).
- `.env` sets the models, providers and port.
