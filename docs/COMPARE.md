# Comparing and combining classifiers

We have two classifiers for type, queue and priority:

| Model | Approach | Strengths | Weaknesses |
|---|---|---|---|
| **SentinelDesk ML** (`scripts/train.py`) | TF-IDF and/or multilingual embeddings feeding a calibrated linear model. The best of three candidates is picked on the dev split. | About 10 ms per ticket, free, deterministic, calibrated confidence, learns the dataset's own labelling habits | No real understanding of the text; weak on rare queues |
| **Gemini "Decision 1"** (teammate) | The LLM reads the ticket and outputs labels + confidence in one request | Understands meaning, impact and negation; good on rare or odd tickets | About 1 s per ticket, costs money, confidence is self-reported (not calibrated), doesn't know the dataset's labelling habits |

The fair way to compare them is the **same test tickets and the same metrics**, then try combining them.

## 1. Use the same split

`splits/split_assignment.csv` is committed. It maps every row of
`aa_dataset-tickets-multi-lang-5-2-50-version.csv` (0-based `row`, header excluded) to a `ticket_id` and a `split`.

- Run Gemini **only on rows where `split == "test"`** (5,7xx tickets), or on a fixed sample of them.
- The prompt may contain **subject + body only**. It must not include the answer, labels or tags.
- Few-shot examples may come **only from `split == "train"`** rows.

## 2. Save predictions in the shared format

Save a CSV to `reports/predictions/<model_name>_test.csv` with one row per test ticket:

| column | required | example |
|---|---|---|
| `ticket_id` | yes | `TKT-01234` |
| `pred_type`, `pred_queue`, `pred_priority` | yes | `Incident`, `IT Support`, `high` |
| `conf_type`, `conf_queue`, `conf_priority` | yes | `0.92` (0–1) |
| `critical_score` | optional | `0.81` = P(high-priority Incident) |
| `p_<head>__<class>` | optional | `p_queue__IT Support` = 0.62 (full probability vector, enables better ensembling) |

Label spellings must match the dataset exactly. For example, `Billing and Payments`, not `Billing`.

## 3. Compare and ensemble

```powershell
py scripts/compare.py
```

This scores every file in `reports/predictions/` on the test labels: macro-F1 per head, high-priority incident recall, latency if a `latency_ms` column exists, and agreement between the models. It also tries three combinations:

1. **Confidence cascade:** keep the ML label when its confidence is above the head's threshold. Otherwise use Gemini. Cheap: Gemini only runs on the uncertain share.
2. **Probability average:** average the class probabilities when both models provide them.
3. **Safety OR:** page if *either* model says high-priority Incident. This maximises incident recall, and the script shows how many extra tickets it flags.

The winner per head becomes what we demo. The cascade is usually the best story: fast and free for most tickets, with the LLM only on the hard ones.
