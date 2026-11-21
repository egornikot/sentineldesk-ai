"""SentinelDesk+ : the combined version.

Built ON TOP of the original `sentineldesk` package (which is left untouched) and adds the best ideas from
the three team builds:
  - ours (sentineldesk):  calibrated classifiers, 3-route gate, quote-verified citations, leakage-aware split
  - twin vote (new):      blend classifier probabilities with a similarity-weighted vote of the nearest
                          TRAIN tickets, tuned on dev (the dataset rewrites each scenario many times)
  - temperature scaling:  per-head calibration on dev (Helios v3, akshita branch)
  - outage burst radar:   >= 3 similar incident tickets in a short window -> one major incident (Helios)
  - PII "what left":      shows exactly the masked text the LLM received (Helios)
  - placeholder cleanup:  dataset "<tel_num>"/"<name>" placeholders normalised in precedents (akshita branch)
  - precedent threshold:  min_similarity where the nearest precedent has the right queue >= 85% (akshita branch)
  - extra metrics:        null-subject slice, cross-lingual retrieval, paraphrase-grouped "honest" split
"""
