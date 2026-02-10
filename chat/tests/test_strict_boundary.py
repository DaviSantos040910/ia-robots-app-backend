from django.test import TestCase
from unittest.mock import MagicMock, patch
from chat.services.strict_boundary import strict_boundary, ResponseMode
from chat.services.evidence_gate import EvidenceDecision

class StrictBoundaryTest(TestCase):
    @patch('chat.services.strict_boundary.evidence_gate.evaluate')
    @patch('chat.services.strict_boundary.vector_service.search_context')
    @patch('chat.services.strict_boundary.intent_service.is_sources_list_intent')
    def test_list_sources_intent(self, mock_intent, mock_search, mock_evidence):
        mock_intent.return_value = True

        mode, _, _, _ = strict_boundary.decide_response_mode("quais fontes?", True, False, 1, 1, [])
        self.assertEqual(mode, ResponseMode.LIST_SOURCES)

    @patch('chat.services.strict_boundary.evidence_gate.evaluate')
    @patch('chat.services.strict_boundary.vector_service.search_context')
    @patch('chat.services.strict_boundary.intent_service.is_sources_list_intent')
    def test_strict_answer(self, mock_intent, mock_search, mock_evidence):
        mock_intent.return_value = False
        mock_search.return_value = ([{'score': 0.2}], [])
        mock_evidence.return_value = (EvidenceDecision.ANSWER, "reason")

        mode, _, _, _ = strict_boundary.decide_response_mode("q?", True, False, 1, 1, [])
        self.assertEqual(mode, ResponseMode.STRICT_ANSWER_WITH_CONTEXT)

    @patch('chat.services.strict_boundary.evidence_gate.evaluate')
    @patch('chat.services.strict_boundary.vector_service.search_context')
    @patch('chat.services.strict_boundary.intent_service.is_sources_list_intent')
    def test_strict_refusal(self, mock_intent, mock_search, mock_evidence):
        mock_intent.return_value = False
        mock_search.return_value = ([{'score': 0.9}], [])
        mock_evidence.return_value = (EvidenceDecision.REFUSE, "reason")

        mode, _, _, _ = strict_boundary.decide_response_mode("q?", True, False, 1, 1, [])
        self.assertEqual(mode, ResponseMode.STRICT_REFUSAL)

    def test_non_strict_web(self):
        mode, _, _, _ = strict_boundary.decide_response_mode("q?", False, True, 1, 1, [])
        self.assertEqual(mode, ResponseMode.NON_STRICT_WEB_OR_GENERAL)

    def test_non_strict_general(self):
        mode, _, _, _ = strict_boundary.decide_response_mode("q?", False, False, 1, 1, [])
        self.assertEqual(mode, ResponseMode.NON_STRICT_WEB_OR_GENERAL)
