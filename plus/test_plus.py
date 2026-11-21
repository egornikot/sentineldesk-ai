"""Offline tests for the twin vote, calibration, burst radar and PII view.   py -m unittest plus/test_plus.py"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PLUS = Path(__file__).resolve().parent
sys.path[:0] = [str(PLUS.parent), str(PLUS)]

import numpy as np  # noqa: E402

from sentineldesk_plus import core  # noqa: E402
from sentineldesk_plus.answers import clean_answer, raw_placeholders  # noqa: E402
from sentineldesk_plus.burst import BurstRadar  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestAnswers(unittest.TestCase):
    def test_placeholders_normalised(self):
        out = clean_answer("Call <tel_num> or ask <Name>, Kontonummer <Kontonummer>.<br>Thanks")
        self.assertIn("[phone number]", out)
        self.assertIn("[name]", out)
        self.assertIn("[account number]", out)
        self.assertNotIn("<", out)

    def test_raw_placeholder_detected(self):
        self.assertEqual(raw_placeholders("Hello <name>, done"), ["<name>"])
        self.assertEqual(raw_placeholders("Hello [name]"), [])


class TestBurst(unittest.TestCase):
    def test_three_similar_incidents_make_one_burst(self):
        r = BurstRadar(min_similarity=0.8, min_size=3)
        base = unit([1, 0.1, 0])
        self.assertIsNone(r.observe(base, "VPN down", "", "en", True))
        self.assertIsNone(r.observe(unit([1, 0.12, 0]), "VPN outage", "", "en", True))
        b = r.observe(unit([1, 0.08, 0.01]), "VPN-Ausfall", "", "de", True)
        self.assertIsNotNone(b)
        self.assertEqual(b["count"], 3)
        self.assertEqual(b["languages"], ["de", "en"])

    def test_unrelated_or_non_incident_does_not_trigger(self):
        r = BurstRadar(min_similarity=0.8, min_size=3)
        for _ in range(3):
            self.assertIsNone(r.observe(unit([1, 0, 0]), "invoice", "", "en", False))
        self.assertIsNone(r.observe(unit([0, 1, 0]), "other", "", "en", True))


class TestTwinVote(unittest.TestCase):
    def test_knn_and_blend(self):
        labels = np.array(["high", "low", "high"], dtype=object)
        idx = np.array([[0, 1, 2]])
        sims = np.array([[0.99, 0.70, 0.95]], dtype=np.float32)
        K = core.knn_proba(idx, sims, labels, ["high", "low"], 3, 0.02)
        self.assertGreater(K[0, 0], 0.9)
        P = core.blend(np.array([[0.2, 0.8]]), K, 0.5)
        self.assertAlmostEqual(float(P.sum()), 1.0, places=6)
        self.assertEqual(int(P.argmax()), 0)

    def test_temperature_keeps_argmax(self):
        P = np.array([[0.6, 0.3, 0.1]])
        Q = core.blend(P, P, 0.0, 0.5)
        self.assertEqual(int(Q.argmax()), 0)
        self.assertGreater(Q[0, 0], 0.6)


if __name__ == "__main__":
    unittest.main()
