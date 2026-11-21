# SentinelDesk

An AI incident triage copilot for customer support. For every ticket it:

1. **Classifies** the ticket's type, queue and priority, each with a calibrated confidence score.
2. **Decides whether a human is needed**: page someone now, send to human review, or mark an AI draft as ready. Every decision lists its reasons.
3. **Retrieves** the 5 most similar resolved tickets from the historical data, with similarity scores. Matching works across languages.
4. **Drafts a reply** in English or German, matching the customer. Every instruction in the reply cites the past ticket it comes from, and the code checks each citation before the draft is shown.
5. **Detects outage bursts**: 3 or more similar incident reports (EN or DE) within 2 hours become one major incident, and new matching tickets go straight to on-call.

> **Core idea:** urgency and evidence are two separate decisions. A hospital VPN outage is paged to an engineer immediately, even when there is no known fix. A routine billing ticket can go to the right queue with high confidence while the AI declines to promise a refund that no past case supports.

**Live demo:** https://egornikot.github.io/sentineldesk-ai/ (type any English or German ticket while the live server is on; otherwise it replays recorded real runs).

## Results on the held-out test set (5,718 tickets never seen in training)

| Metric | Result |
|---|---|
| Macro-F1: type / queue / priority | **0.929** / **0.782** / **0.818** |
| High-priority incident recall | **91.5%**, with 16.5% of other tickets paged |
| Calibration error (ECE, type) | **0.013**: a confidence of 0.8 really means about 80% correct |
| Routes: page now / human review / draft ready | 29.5% / 16.5% / **54.0%** |
| Cross-lingual precedent in top 5 | 63.5% of queries |
| Grounding (live LLM, demo cases) | 100% of cited claims verified against the retrieved tickets |
| Latency | classify + retrieve ≈ 120 ms; end to end ≈ 1.3 s with Gemini 3.5 Flash-Lite |

**Ablation.** Without the twin vote and calibration (base classifiers only), macro-F1 drops to 0.859 / 0.571 / 0.587, 26.6% of other tickets get paged, and only 9% of tickets reach draft ready.

**The data finding.** On the standard split, 42% of test tickets have a near-copy (cosine ≥ 0.95) in training, with the same priority 98% of the time. We therefore also report an **honest split** where rewrites of one scenario are grouped (0% near-copies). There: type F1 0.815, queue 0.367, priority 0.458, incident recall 96.8%. That is the number to expect on genuinely new tickets. Full table: [plus/reports/COMPARE_RESULTS.md](plus/reports/COMPARE_RESULTS.md).

**No leakage.** Duplicate texts are grouped, then split 72/8/20 (train/dev/test) and stratified by type and priority. Models, blend weights and thresholds are chosen on dev only. Only train tickets are indexed for retrieval. The exact assignment is in `splits/split_assignment.csv` (see [docs/COMPARE.md](docs/COMPARE.md)).

## Pipeline

```
ticket (subject + body)
  → preprocess   mask PII (email, phone, IP, IBAN, cards, accounts, keys), detect EN/DE
  → classify     calibrated classifiers per head
  → twin vote    the 5 most similar train tickets vote; blended with the classifier (weights tuned on dev)
  → retrieve     top-5 similar resolved TRAIN tickets, multilingual embeddings
  → burst radar  ≥3 similar incident reports within 2 h → one major incident MIR-…, escalated
  → gate         PAGE NOW / HUMAN REVIEW / DRAFT READY, with reasons
  → generate     LLM reply as JSON claims, each with a ticket ID + verbatim excerpt
  → verify       each excerpt must fuzzy-match the cited ticket; flag refunds/amounts/SLAs/legal promises
  → dashboard    full trace, per-stage latency, evidence, and the exact masked text sent to the LLM
```

**Gate rules** (plain code, never the LLM).
- **Page now:** a confident Incident + high label, a critical score over the tuned threshold, severe impact words (clinical, outage, security, data loss, with urgency or a second signal), or membership of a detected outage burst.
- **Human review:** low confidence on any head, weak precedent (the best past match is below the similarity where its queue is right ≥85% of the time), a request for a refund or other commitment, or a draft that fails verification.
- **Draft ready:** everything else. The draft can be sent with one click.

**LLM.** Gemini 3.5 Flash-Lite by default. If it fails or takes longer than 12 s, Claude takes over (when `ANTHROPIC_API_KEY` is set). With no key, an offline mock drafts from the top precedent. Personal data is masked before any model sees the text.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
copy .env.example .env        # add GEMINI_API_KEY and/or ANTHROPIC_API_KEY (or leave empty: mock mode)
py scripts/prepare_data.py    # download, clean, dedupe, grouped split
py scripts/train.py           # base classifiers + thresholds + retrieval index (~10 min first time)
py plus/train_plus.py         # twin vote + temperature + thresholds on dev (~2 min)
py plus/evaluate_plus.py      # test metrics -> plus/reports/
py run.py                     # console at http://localhost:8000 (or double-click start.bat)
```

Public demo from your own computer: double-click `go_public.bat`. It opens a free Cloudflare tunnel and points the GitHub Pages dashboard at it; keep the window open.

Checks:

```powershell
py -m unittest discover -s tests       # offline tests
py plus/test_plus.py                   # twin vote, calibration, burst radar tests
py evals/run_evals.py --live           # demo + safety cases with the real LLM
```

## Code layout

- `sentineldesk/`: base stages (PII masking, classifiers, retrieval, gate, LLM drafting, citation check, server).
- `plus/`: the model layer on top (twin vote, calibration, burst radar, PII view, evaluation). See [plus/README.md](plus/README.md).
- `scripts/`: data preparation, training, public demo (`go_public.py`). `evals/`: demo and safety cases.
