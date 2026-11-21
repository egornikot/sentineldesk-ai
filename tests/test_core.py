"""Offline unit tests (no models, no API key needed).   py -m unittest discover -s tests"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentineldesk import gate, guard, rules, verify  # noqa: E402
from sentineldesk.classify import Classifier  # noqa: E402
from sentineldesk.config import Settings  # noqa: E402
from sentineldesk.generate import draft_reply  # noqa: E402
from sentineldesk.llm import MockLLM  # noqa: E402
from sentineldesk.pipeline import Pipeline, detect_language  # noqa: E402
from sentineldesk.retrieve import Retriever  # noqa: E402

TH = {"critical": 0.4, "min_confidence": 0.55, "min_similarity": 0.5}
CONFIDENT = {"type": {"label": "Request", "confidence": 0.9}, "queue": {"label": "Billing and Payments", "confidence": 0.9},
             "priority": {"label": "low", "confidence": 0.9}, "critical_score": 0.05}
CASE = [{"ticket_id": "TKT-00001", "similarity": 0.8, "answer": "Please clear your browser cache and sign in again. This resolves most login loops."}]


class Guard(unittest.TestCase):
    def test_masks_contact_data(self):
        r = guard.redact("Mail me at a.b@example.com or call +49 170 1234567, IP 10.0.0.12, account number 12345678")
        for kind in ("email", "phone", "ip_address", "account_number"):
            self.assertIn(kind, r.kinds)
        self.assertNotIn("example.com", r.text)

    def test_keeps_technical_text(self):
        self.assertFalse(guard.redact("Error 500 on version 2.4.1 after the update").changed)


class Rules(unittest.TestCase):
    def test_impact_and_negation(self):
        self.assertIn("clinical", rules.impact_hits("Doctors cannot open patient records"))
        self.assertNotIn("urgency", rules.impact_hits("This is not urgent at all"))

    def test_promises(self):
        self.assertEqual(set(rules.promise_hits("guarantee a $500 refund")), {"refund", "amount", "guarantee"})


class Gate(unittest.TestCase):
    def test_hospital_outage_pages(self):
        d = gate.decide("VPN outage, doctors cannot access patient records, urgent", CONFIDENT, CASE, TH)
        self.assertEqual(d["route"], "page_now")

    def test_predicted_high_incident_pages(self):
        cls = {**CONFIDENT, "type": {"label": "Incident", "confidence": 0.8}, "priority": {"label": "high", "confidence": 0.7}}
        self.assertEqual(gate.decide("printer broken", cls, CASE, TH)["route"], "page_now")

    def test_refund_promise_goes_to_review(self):
        self.assertEqual(gate.decide("guarantee us a $500 refund", CONFIDENT, CASE, TH)["route"], "human_review")

    def test_weak_evidence_goes_to_review(self):
        weak = [{**CASE[0], "similarity": 0.2}]
        self.assertEqual(gate.decide("how do I export a report", CONFIDENT, weak, TH)["route"], "human_review")

    def test_clean_ticket_is_draft_ready(self):
        self.assertEqual(gate.decide("how do I export a report", CONFIDENT, CASE, TH)["route"], "draft_ready")


class Verify(unittest.TestCase):
    def test_supported_claim_passes(self):
        draft = {"reply": "Clear your cache [TKT-00001]", "claims": [
            {"sentence": "Clear your cache.", "ticket_id": "TKT-00001", "excerpt": "clear your browser cache and sign in again"}]}
        self.assertTrue(verify.verify(draft, CASE)["ok"])

    def test_invented_ticket_and_promise_fail(self):
        draft = {"reply": "We guarantee a full refund.", "claims": [
            {"sentence": "Reinstall.", "ticket_id": "TKT-99999", "excerpt": "reinstall the app"}]}
        v = verify.verify(draft, CASE)
        self.assertFalse(v["ok"])
        self.assertEqual(v["grounding_rate"], 0.0)


class Pipeline_(unittest.TestCase):
    def test_runs_end_to_end_without_models(self):
        s = Settings()
        s.llm_mode = "mock"
        pipe = Pipeline(s, Classifier(), Retriever(), MockLLM())
        r = pipe.run("", "Our VPN is down for all users and clinicians cannot reach patient records!")
        self.assertEqual(r["decision"]["route"], "page_now")
        self.assertIn("total", r["timings_ms"])

    def test_mock_draft_is_grounded(self):
        d = draft_reply(MockLLM(), {"subject": "", "body": "login loop", "language": "en"}, CASE, {"route": "draft_ready"})
        self.assertTrue(verify.verify(d, CASE)["ok"])

    def test_language(self):
        self.assertEqual(detect_language("Hallo, ich kann mich nicht anmelden und bitte um Hilfe."), "de")
        self.assertEqual(detect_language("Hello, I cannot log in to my account, please help."), "en")


if __name__ == "__main__":
    unittest.main()
