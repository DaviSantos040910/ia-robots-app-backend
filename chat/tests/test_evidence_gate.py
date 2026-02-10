from django.test import TestCase
from chat.services.evidence_gate import evidence_gate, EvidenceDecision

class EvidenceGateTest(TestCase):
    def test_empty_refuse(self):
        decision, reason = evidence_gate.evaluate([], 0.45)
        self.assertEqual(decision, EvidenceDecision.REFUSE)
        self.assertEqual(reason, "no_context")

    def test_strong_match_answer(self):
        # Best score 0.2 < 0.45 * 0.92 (~0.41)
        docs = [{'score': 0.2, 'source_id': '1'}]
        decision, reason = evidence_gate.evaluate(docs, 0.45)
        self.assertEqual(decision, EvidenceDecision.ANSWER)
        self.assertIn("strong_match", reason)

    def test_weak_match_refuse(self):
        # Best score 0.6 > 0.45 * 1.08 (~0.48)
        docs = [{'score': 0.6, 'source_id': '1'}]
        decision, reason = evidence_gate.evaluate(docs, 0.45)
        self.assertEqual(decision, EvidenceDecision.REFUSE)
        self.assertIn("weak_match", reason)

    def test_gray_zone_multi_source(self):
        # Best 0.46 (gray), but 2 unique sources
        docs = [
            {'score': 0.46, 'source_id': '1'},
            {'score': 0.47, 'source_id': '2'}
        ]
        decision, reason = evidence_gate.evaluate(docs, 0.45)
        self.assertEqual(decision, EvidenceDecision.ANSWER)
        self.assertIn("gray_zone_multi_source", reason)

    def test_gray_zone_short_text_refuse(self):
        # Best 0.46 (gray), 1 source, very short text
        docs = [{'score': 0.46, 'source_id': '1', 'content': 'Too short'}]
        decision, reason = evidence_gate.evaluate(docs, 0.45)
        self.assertEqual(decision, EvidenceDecision.REFUSE)
        self.assertIn("gray_zone_short_text", reason)

    def test_gray_zone_consistent_answer(self):
        # Best 0.44 (grayish but close), Avg low, Gap low
        # Threshold 0.45. Avg needs <= 0.45. Gap <= 0.03.
        # 0.44 and 0.45 -> Avg 0.445 (<=0.45), Gap 0.01 (<=0.03)
        # Assuming only 1 source ID to bypass multi-source check
        long_text = "a" * 250
        docs = [
            {'score': 0.44, 'source_id': '1', 'content': long_text},
            {'score': 0.45, 'source_id': '1', 'content': long_text}
        ]
        decision, reason = evidence_gate.evaluate(docs, 0.45)
        self.assertEqual(decision, EvidenceDecision.ANSWER)
        self.assertIn("gray_zone_consistent", reason)
