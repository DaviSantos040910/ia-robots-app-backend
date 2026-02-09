from django.test import TestCase
from unittest.mock import MagicMock
from chat.services.chat_service import _detect_lang, _safe_excerpt, _build_strict_refusal

class StrictRefusalLogicTest(TestCase):
    def test_detect_lang_heuristics(self):
        """Test language detection heuristics."""
        # "Qual" might not be in the list, let's verify markers
        # pt_markers = ["você", "não", "por que", "onde", "fonte", "fontes", "documento", "documentos", "tutor", "quais", "quem"]

        self.assertEqual(_detect_lang("Quem é você?"), "pt")
        self.assertEqual(_detect_lang("Onde fica isso?"), "pt")
        self.assertEqual(_detect_lang("fonte dos dados"), "pt")

        self.assertEqual(_detect_lang("¿Donde esta?"), "es")
        self.assertEqual(_detect_lang("¡Hola!"), "es")
        self.assertEqual(_detect_lang("por qué no?"), "es")
        self.assertEqual(_detect_lang("fuente"), "es")

        self.assertEqual(_detect_lang("Where is it?"), "en")
        self.assertEqual(_detect_lang("What is this?"), "en")
        self.assertEqual(_detect_lang("Random text"), "en")

    def test_safe_excerpt(self):
        """Test excerpt truncation and cleaning."""
        text = "Line 1\nLine 2   with   spaces"
        self.assertEqual(_safe_excerpt(text), "Line 1 Line 2 with spaces")

        long_text = "a" * 200
        excerpt = _safe_excerpt(long_text, max_len=10)
        self.assertEqual(excerpt, "aaaaaaaaaa…")

    def test_build_strict_refusal_pt(self):
        """Test refusal construction in Portuguese."""
        # Case 1: Has sources
        msg = _build_strict_refusal("TutorBot", "Como fazer bolo?", lang="pt", has_any_sources=True)
        self.assertIn("TutorBot: Não encontrei essa informação", msg)
        self.assertIn("Pergunta: “Como fazer bolo?”", msg)
        self.assertIn("adicionar uma fonte relevante", msg)

        # Case 2: No sources
        msg = _build_strict_refusal(None, "Como fazer bolo?", lang="pt", has_any_sources=False)
        self.assertIn("No modo restrito, eu só posso responder usando fontes.", msg) # Prefix is empty
        self.assertIn("Pergunta: “Como fazer bolo?”", msg)
        self.assertIn("adicione uma fonte (PDF", msg)

    def test_build_strict_refusal_en(self):
        """Test refusal construction in English."""
        # Case 1: Has sources
        msg = _build_strict_refusal("Bot", "Why?", lang="en", has_any_sources=True)
        self.assertIn("Bot: I couldn’t find this information", msg)

        # Case 2: No sources
        msg = _build_strict_refusal(None, "Why?", lang="en", has_any_sources=False)
        self.assertIn("In strict mode, I can only answer using sources.", msg)

    def test_build_strict_refusal_es(self):
        """Test refusal construction in Spanish."""
        # Case 1: Has sources
        msg = _build_strict_refusal("Bot", "¿Por qué?", lang="es", has_any_sources=True)
        self.assertIn("Bot: No encontré esta información", msg)

        # Case 2: No sources
        msg = _build_strict_refusal(None, "¿Por qué?", lang="es", has_any_sources=False)
        self.assertIn("En modo estricto, solo puedo responder usando fuentes.", msg)
