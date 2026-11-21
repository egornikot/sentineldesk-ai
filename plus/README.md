# plus/: the model layer on top of the base pipeline

`sentineldesk/` holds the base stages (PII masking, classifiers, retrieval, gate, LLM drafting, citation check).
`plus/` adds the layer that makes SentinelDesk accurate and safe enough to present. `py run.py` uses both.

| Component | What it does |
|---|---|
| **Twin vote** | The dataset rewrites each scenario many times. The 5 most similar *training* tickets vote on type/queue/priority/critical; the vote is blended with the classifier (weights tuned on dev). |
| **Temperature scaling** | Per-head calibration on dev, so an 80% confidence means about 80% correct. |
| **Outage burst radar** | ≥3 similar incident reports (EN + DE) within 2 h become one major incident `MIR-…`; new matching tickets are escalated automatically. "Simulate outage burst" button on the dashboard. |
| **"What left your network"** | Shows the exact masked text the LLM received (`[EMAIL]`, `[PHONE]`, …). |
| **Placeholder cleanup** | The dataset's `<tel_num>`, `<name>`, `<Tel-Nummer>` … are normalised in precedents; a raw placeholder in a draft is flagged. |
| **Precedent threshold by queue agreement** | "Weak precedent" = below the similarity where the nearest past ticket has the right queue ≥85% of the time. |
| **Honest split + extra metrics** | Paraphrase-grouped split (no rewrite of a test scenario in train), null-subject slice, cross-lingual retrieval share. |

## Results

See [reports/COMPARE_RESULTS.md](reports/COMPARE_RESULTS.md): SentinelDesk on the held-out test tickets, plus an
ablation with the base classifiers only, on both the standard and the honest split.

**The data finding behind it.** On the standard split, 42% of test tickets have a near-copy (cosine ≥ 0.95) in
training, and that near-copy has the same priority 98% of the time. A random or exact-duplicate split therefore
rewards memorising rewrites. We report both:
- **standard split**: comparable with other teams' numbers;
- **honest split**: rewrites grouped (0% near-copies), the number that predicts performance on genuinely new tickets.

## Commands

```powershell
py plus/train_plus.py            # tune twin vote + temperature + thresholds on dev  (~2 min, uses models/)
py plus/evaluate_plus.py         # score on test -> plus/reports/
py plus/build_static_plus.py     # record real runs for the GitHub Pages fallback -> docs/
```

Honest split (optional, ~15 min): `py plus/build_honest_split.py`, `py plus/train_honest.py`,
`py plus/train_plus.py --models plus/honest/models --split plus/honest/processed --out plus/honest/models/plus.json`.
